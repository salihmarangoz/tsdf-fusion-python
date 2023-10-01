# Copyright (c) 2018 Andy Zeng

import numpy as np

from numba import njit, prange
from skimage import measure

try:
  import pycuda.driver as cuda
  import pycuda.autoinit
  from pycuda.compiler import SourceModule
except Exception as err:
  print('Warning: {}'.format(err))
  print('Failed to import PyCUDA. Running fusion in CPU mode.')


class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_bnds, voxel_size):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    # Define voxel volume parameters
    self._vol_bnds = vol_bnds
    self._voxel_size = float(voxel_size)
    self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
    self._color_const = 256 * 256

    # Adjust volume bounds and ensure C-order contiguous
    self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
    self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
    self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)

    print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
      self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
      self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
    )

    # Initialize pointers to voxel volume in CPU memory
    self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
    # for computing the cumulative moving average of observations per voxel
    self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
    self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

    # Copy voxel volumes to GPU
    self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
    cuda.memcpy_htod(self._tsdf_vol_gpu,self._tsdf_vol_cpu)
    self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
    cuda.memcpy_htod(self._weight_vol_gpu,self._weight_vol_cpu)
    self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
    cuda.memcpy_htod(self._color_vol_gpu,self._color_vol_cpu)

    self._cuda_integrate = None

    # Determine block/grid size on GPU
    gpu_dev = cuda.Device(0)
    self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
    n_blocks = int(np.ceil(float(np.prod(self._vol_dim))/float(self._max_gpu_threads_per_block)))
    grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,int(np.floor(np.cbrt(n_blocks))))
    grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,int(np.floor(np.sqrt(n_blocks/grid_dim_x))))
    grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,int(np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
    self._max_gpu_grid_dim = np.array([grid_dim_x,grid_dim_y,grid_dim_z]).astype(int)
    self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim))/float(np.prod(self._max_gpu_grid_dim)*self._max_gpu_threads_per_block)))

  def init_cuda_function(self):
    if self._cuda_integrate is not None: return

    # Cuda kernel function (C++)
    cuda_src = """
      __global__ void integrate(float * tsdf_vol,
                                float * weight_vol,
                                float * color_vol,
                                float * vol_origin,
                                float * cam_intr,
                                float * cam_pose,
                                float * other_params,
                                float * color_im,
                                float * depth_im) {{
        // Get constant parameters
        int vol_dim_x = {vol_dim_x};
        int vol_dim_y = {vol_dim_y};
        int vol_dim_z = {vol_dim_z};
        float voxel_size = {voxel_size};
        int im_h = {im_h};
        int im_w = {im_w};
        float trunc_margin = {trunc_margin};
        float obs_weight = {obs_weight};
        float max_depth = {max_depth};
        float cam_intr_0 = {cam_intr_0};
        float cam_intr_2 = {cam_intr_2};
        float cam_intr_4 = {cam_intr_4};
        float cam_intr_5 = {cam_intr_5};
        
        // Get voxel index
        int gpu_loop_idx = (int) other_params[0];
        int max_threads_per_block = blockDim.x;
        int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
        int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block + block_idx*max_threads_per_block + threadIdx.x;
        if (voxel_idx >= vol_dim_x*vol_dim_y*vol_dim_z)
            return;

        // Get voxel grid coordinates (note: be careful when casting)
        int voxel_x = voxel_idx/(vol_dim_y*vol_dim_z);
        int voxel_y = (((voxel_idx-voxel_x*vol_dim_y*vol_dim_z))/vol_dim_z);
        int voxel_z = voxel_idx-voxel_x*vol_dim_y*vol_dim_z-voxel_y*vol_dim_z;

        // Voxel grid coordinates to world coordinates
        
        float pt_x = vol_origin[0]+voxel_x*voxel_size;
        float pt_y = vol_origin[1]+voxel_y*voxel_size;
        float pt_z = vol_origin[2]+voxel_z*voxel_size;

        // World coordinates to camera coordinates
        float tmp_pt_x = pt_x-cam_pose[0*4+3];
        float tmp_pt_y = pt_y-cam_pose[1*4+3];
        float tmp_pt_z = pt_z-cam_pose[2*4+3];
        float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x + cam_pose[1*4+0]*tmp_pt_y + cam_pose[2*4+0]*tmp_pt_z;
        float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x + cam_pose[1*4+1]*tmp_pt_y + cam_pose[2*4+1]*tmp_pt_z;
        float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x + cam_pose[1*4+2]*tmp_pt_y + cam_pose[2*4+2]*tmp_pt_z;

        // Camera coordinates to image pixels
        //int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
        //int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
        int pixel_x = (int) roundf(cam_intr_0*(cam_pt_x/cam_pt_z)+cam_intr_2);
        int pixel_y = (int) roundf(cam_intr_4*(cam_pt_y/cam_pt_z)+cam_intr_5);
        
        // Skip if outside view frustum
        if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
            return;

        // Skip invalid depth
        float depth_value = depth_im[pixel_y*im_w + pixel_x];
        if (depth_value == 0 || max_depth < depth_value)
            return;

        // Integrate TSDF
        float depth_diff = depth_value-cam_pt_z;
        if (depth_diff < -trunc_margin)
            return;
        float dist = -fmin(1.0f,depth_diff/trunc_margin);
        float w_old = weight_vol[voxel_idx];
        float w_new = w_old + obs_weight;
        weight_vol[voxel_idx] = w_new;
        tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;

        // Integrate color
        float old_color = color_vol[voxel_idx];
        float old_b = floorf(old_color/(256*256));
        float old_g = floorf((old_color-old_b*256*256)/256);
        float old_r = old_color-old_b*256*256-old_g*256;
        float new_color = color_im[pixel_y*im_w+pixel_x];
        float new_b = floorf(new_color/(256*256));
        float new_g = floorf((new_color-new_b*256*256)/256);
        float new_r = new_color-new_b*256*256-new_g*256;

        // Fast method for colors
        new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
        new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
        new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);

        // More accurate method for colors
        // new_b = sqrt((old_b*old_b*w_old + new_b*new_b*obs_weight)/w_new);
        // new_g = sqrt((old_g*old_g*w_old + new_g*new_g*obs_weight)/w_new);
        // new_r = sqrt((old_r*old_r*w_old + new_r*new_r*obs_weight)/w_new);
        // new_b = fmin(roundf(new_b),255.0f);
        // new_g = fmin(roundf(new_g),255.0f);
        // new_r = fmin(roundf(new_r),255.0f);

        color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
      }}""".format(voxel_size = self._voxel_size,
                   trunc_margin = self._trunc_margin,
                   vol_dim_x = self._vol_dim[0],
                   vol_dim_y = self._vol_dim[1],
                   vol_dim_z = self._vol_dim[2],
                   im_h = self._im_h,
                   im_w = self._im_w,
                   obs_weight = self._obs_weight,
                   max_depth = 5.0,
                   cam_intr_0 = self._cam_intr[0],
                   cam_intr_1 = self._cam_intr[1],
                   cam_intr_2 = self._cam_intr[2],
                   cam_intr_3 = self._cam_intr[3],
                   cam_intr_4 = self._cam_intr[4],
                   cam_intr_5 = self._cam_intr[5],
                   cam_intr_6 = self._cam_intr[6],
                   cam_intr_7 = self._cam_intr[7],
                   cam_intr_8 = self._cam_intr[8],) 
    self._cuda_src_mod = SourceModule(cuda_src, options=["--use_fast_math", "-O3", "--extra-device-vectorization"])
    self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

  def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign for the current observation. A higher
        value
    """
    self._im_h, self._im_w = depth_im.shape
    self._cam_intr = cam_intr.reshape(-1).astype(np.float32)
    self._obs_weight = obs_weight
    self.init_cuda_function()

    # Fold RGB color image into a single channel image
    color_im = color_im.astype(np.float32)
    color_im = np.floor(color_im[...,2]*self._color_const + color_im[...,1]*256 + color_im[...,0])

    for gpu_loop_idx in range(self._n_gpu_loops):
      self._cuda_integrate(self._tsdf_vol_gpu,
                          self._weight_vol_gpu,
                          self._color_vol_gpu,
                          cuda.In(self._vol_origin.astype(np.float32)), #dynamic
                          cuda.In(self._cam_intr.reshape(-1).astype(np.float32)), #done
                          cuda.In(cam_pose.reshape(-1).astype(np.float32)), #dynamic
                          cuda.In(np.asarray([
                            gpu_loop_idx,
                            self._voxel_size, #done
                            self._im_h, #done
                            self._im_w, #done
                            self._trunc_margin, #done
                            self._obs_weight #done
                          ], np.float32)),
                          cuda.In(color_im.reshape(-1).astype(np.float32)),
                          cuda.In(depth_im.reshape(-1).astype(np.float32)),
                          block=(self._max_gpu_threads_per_block,1,1),
                          grid=(
                            int(self._max_gpu_grid_dim[0]),
                            int(self._max_gpu_grid_dim[1]),
                            int(self._max_gpu_grid_dim[2]),
                          )
      )

  def get_volume(self):
    cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
    cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
    return self._tsdf_vol_cpu, self._color_vol_cpu

  def get_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """
    tsdf_vol, color_vol = self.get_volume()

    # Marching cubes
    verts = measure.marching_cubes(tsdf_vol, mask=np.logical_and(tsdf_vol > -0.5,tsdf_vol < 0.5), level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + self._vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._color_const)
    colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
    colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    pc = np.hstack([verts, colors])
    return pc

  def get_mesh(self):
    """Compute a mesh from the voxel volume using marching cubes.
    """
    tsdf_vol, color_vol = self.get_volume()

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, mask=np.logical_and(tsdf_vol > -0.5,tsdf_vol < 0.5), level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size+self._vol_origin  # voxel grid coordinates to world coordinates

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
    colors_b = np.floor(rgb_vals/self._color_const)
    colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
    colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
    colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
    colors = colors.astype(np.uint8)
    return verts, faces, norms, colors


def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()


def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))
