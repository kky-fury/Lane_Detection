#Inspired from KITTI Road DataSet - transform2BEV
import numpy as np
import cv2 as cv
import os

class BevParams(object):
	bev_size = None
	bev_res = None
	bev_xLimits = None
	bev_zLimits = None
	imSize = None
	imSize_back = None

	def __init__(self, bev_res, bev_xLimits, bev_zLimits, imSize):
		bev_size = (round((bev_zLimits[1] - bev_zLimits[0]) / bev_res),round((bev_xLimits[1] - bev_xLimits[0]) / bev_res))
		self.bev_size = bev_size
		self.bev_res = bev_res
		self.bev_xLimits = bev_xLimits
		self.bev_zLimits = bev_zLimits
		self.imSize = imSize


class Calibration(object):
	R0_rect = None
	P2 = None
	Tr33 = None
	Tr = None
	Tr_cam_to_road = None

	def __init__(self):
		pass

	def setup_calib(self, P2, R0_rect, Tr_cam_to_road):
		self.P2 = P2
		R0_rect_raw = R0_rect
		self.R0_rect = np.matrix(np.hstack((np.vstack((R0_rect_raw, np.zeros((1,3), np.float64))), np.zeros((4,1), np.float64))))
		self.R0_rect[3,3]=1.
		R2_1 = np.dot(self.P2, self.R0_rect)

		Tr_cam_to_road_raw = Tr_cam_to_road
		self.Tr_cam_to_road = np.matrix(np.vstack((Tr_cam_to_road_raw, np.zeros((1,4), np.float64))))
		self.Tr_cam_to_road[3,3]=1.
		self.Tr = np.dot(R2_1,  self.Tr_cam_to_road.I)
		self.Tr33 =  self.Tr[:,[0,2,3]]

	def get_matrix33(self):
		assert self.Tr33 is not None
		# print(self.Tr33)
		return self.Tr33






class BirdsEyeView(object):
	imSize = None
	bevParams = None
	invalid_value = float('-INFINITY')
	im_u_float = None
	im_v_float = None
	bev_x_ind = None
	bev_z_ind = None

	def __init__(self, bev_res= 0.1, bev_xRange_minMax = (-10, 10), bev_zRange_minMax = (6, 46)):
		self.calib = Calibration()
		bev_res = bev_res
		bev_xRange_minMax = bev_xRange_minMax
		bev_zRange_minMax = bev_zRange_minMax
		self.bevParams = BevParams(bev_res, bev_xRange_minMax, bev_zRange_minMax, self.imSize)

	def setup(self,P2, R0_rect, Tr_cam_to_road):
		self.calib.setup_calib(P2, R0_rect, Tr_cam_to_road)
		self.set_matrix33(self.calib.get_matrix33())

	def world2image(self,X_world, Y_world, Z_world):
		if not type(Y_world) == np.ndarray:
			Y_world = np.ones_like(Z_world)*Y_world
		y = np.vstack((X_world, Y_world, Z_world, np.ones_like(Z_world)))
		test  = self.world2image_uvMat(np.vstack((X_world, Z_world, np.ones_like(Z_world) )))

		self.xi1 = test[0,:]
		self.yi1 = test[1,:]

		assert  self.imSize != None
		condition = ~((self.yi1 >= 1) & (self.xi1 >= 1) & (self.yi1 <= self.imSize[0]) & (self.xi1 <= self.imSize[1]))
		if isinstance(condition, np.ndarray):
			self.xi1[condition] = self.invalid_value
			self.yi1[condition] = self.invalid_value
		elif condition == True:
			self.xi1 = self.invalid_value
			self.yi1 = self.invalid_value

	def world2image_uvMat(self, uv_mat):
		if uv_mat.shape[0] == 2:
			if len(uv_mat.shape)==1:
				uv_mat = uv_mat.reshape(uv_mat.shape + (1,))
			uv_mat = np.vstack((uv_mat, np.ones((1, uv_mat.shape[1]), uv_mat.dtype)))
		result = np.dot(self.Tr33, uv_mat)
		resultB = np.broadcast_arrays(result, result[-1,:])
		return resultB[0] / resultB[1]

	def computeBEVLookUpTable(self, cropping_ul = None, cropping_size = None):
		mgrid = np.lib.index_tricks.nd_grid()
		res = self.bevParams.bev_res

		x_vec = np.arange(self.bevParams.bev_xLimits[0] + res / 2, self.bevParams.bev_xLimits[1], res)
		z_vec = z_vec = np.arange(self.bevParams.bev_zLimits[1] - res / 2, self.bevParams.bev_zLimits[0], -res)
		XZ_mesh = np.meshgrid(x_vec, z_vec)

		assert XZ_mesh[0].shape == self.bevParams.bev_size

		Z_mesh_vec = (np.reshape(XZ_mesh[1], (self.bevParams.bev_size[0] * self.bevParams.bev_size[1]), order = 'F')).astype('f4')
		X_mesh_vec = (np.reshape(XZ_mesh[0], (self.bevParams.bev_size[0] * self.bevParams.bev_size[1]), order = 'F')).astype('f4')

		self.world2image(X_mesh_vec, 0, Z_mesh_vec)
		if (cropping_ul is not None):
			valid_selector = np.ones((self.bevParams.bev_size[0] * self.bevParams.bev_size[1],), dtype = 'bool')
			valid_selector = valid_selector & (self.yi1 >= cropping_ul[0]) & (self.xi1 >= cropping_ul[1])
			if (cropping_size is not None):
				valid_selector = valid_selector & (self.yi1 <= (cropping_ul[0] + cropping_size[0])) & (self.xi1 <= (cropping_ul[1] + cropping_size[1]))
			selector = (~(self.xi1 == self.invalid_value)).reshape(valid_selector.shape) & valid_selector
		else:
			selector = ~(self.xi1 == self.invalid_value)

		y_OI_im_sel = self.yi1[selector]
		x_OI_im_sel = self.xi1[selector]

		ZX_ind = (mgrid[1:self.bevParams.bev_size[0] + 1, 1:self.bevParams.bev_size[1] + 1]).astype('i4')
		Z_ind_vec = np.reshape(ZX_ind[0], selector.shape, order = 'F')
		X_ind_vec = np.reshape(ZX_ind[1], selector.shape, order = 'F')

		Z_ind_vec_sel = Z_ind_vec[selector]
		X_ind_vec_sel = X_ind_vec[selector]

		self.im_u_float = x_OI_im_sel
		self.im_v_float = y_OI_im_sel
		self.bev_x_ind = X_ind_vec_sel.reshape(x_OI_im_sel.shape)
		self.bev_z_ind = Z_ind_vec_sel.reshape(y_OI_im_sel.shape)

	def transformImage2BEV(self, inImage, out_dtype = 'f4'):

		assert self.im_u_float is not None
		assert self.im_v_float is not None
		assert self.im_v_float is not None
		assert self.bev_z_ind is not None

		if len(inImage.shape) > 2:
		 	outputData = np.zeros(self.bevParams.bev_size + (inImage.shape[2],), dtype = out_dtype)
		 	for channel in range(0, inImage.shape[2]):
		 		outputData[self.bev_z_ind-1, self.bev_x_ind-1, channel] = inImage[self.im_v_float.astype('u4')-1, self.im_u_float.astype('u4')-1, channel]
		else:
		 	outputData = np.zeros(self.bevParams.bev_size, dtype = out_dtype)
		 	outputData[self.bev_z_ind-1, self.bev_x_ind-1] = inImage[self.im_v_float.astype('u4')-1, self.im_u_float.astype('u4')-1]

		return outputData


	def set_matrix33(self, matrix33):
		self.Tr33 = matrix33

	def compute(self, image):
		self.imSize = image.shape
		self.computeBEVLookUpTable()
		return self.transformImage2BEV(image, out_dtype = image.dtype)



input_image = cv.imread("/home/mohak/Downloads/data_road/testing/image_2/umm_000026.png",cv.IMREAD_COLOR)
# input_image = cv.imread("/home/mohak/Downloads/roma/BDXD54/IMG00106.jpg",cv.IMREAD_COLOR)
# cv.imshow("Image",input_image)
# cv.waitKey(0)

bev = BirdsEyeView()

# P2 = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
#               [0.000000000000e+00 ,7.215377000000e+02 ,1.728540000000e+02 ,2.163791000000e-01],
#               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]],dtype=np.float64)

P2 = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
              [ 0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02,2.163791000000e-01],
              [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00,2.745884000000e-03]],dtype=np.float64)

# P2 = np.array([[7.8429542068e+000, 0, 6.3423199573e+002, -2.0547200987e+000],
#               [0,7.8429542068e+000,7.4783942715e+002,4.1145446883e-001],
#               [0,0,1,0]],dtype=np.float64)
R0_rect = np.array([[9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03],
                   [-9.869795000000e-03, 9.999421000000e-01,-4.278459000000e-03],
                   [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01]], dtype = np.float64)
# R0_rect = np.array([[0.9999274,-0.01030258,-0.00624997],
#                    [-0.00893876 ,-0.28634691,-0.95808429],
#                    [0.00808109 ,0.95807064,-0.2864182]],dtype = np.float64)

Tr_cam_to_road = np.array([[9.999286320609e-01, -9.386276237978e-03, -7.395286882057e-03,1.565992649364e-02],
                          [9.334868787004e-03, 9.999322098894e-01, -6.955575928437e-03, -1.578779135636e+00],
                          [7.460072373769e-03, 6.886042919607e-03, 9.999484185327e-01,2.854118763826e-01]], dtype =np.float64)


# print(P2)
# print(R0_rect)
# print(Tr_cam_to_road)
# print(bev.calib)
bev.setup(P2,R0_rect,Tr_cam_to_road)
print(bev.Tr33)
# cv.imwrite("/home/mohak/IPM_test_images/IPM_test_image_3.png",output_image)
# print(output_image)
# cv.imshow("Result",warp)
# cv.waitKey(0)







