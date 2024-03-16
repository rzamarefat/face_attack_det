import tensorflow as tf
import os
import numpy as np
from .attack.model import Generator, Discriminator, region_estimator
from .attack import ckpts_config as ckpt_configs 
import cv2
import gdown
import zipfile

class Attack(object):
	def __init__(self, attack_name="paper", face_det_model=None,ckpt_path=None, device="cpu"):
		self._face_det_model = face_det_model
		self._device = device
		self._attack_name = attack_name

		self.RE  = region_estimator()
		self.gen = Generator(self.RE)
		# self.gen_pretrained = Generator(self.RE)
		# self.disc1 = Discriminator(1,4)
		# self.disc2 = Discriminator(2,4)
		# self.disc3 = Discriminator(4,4)

		self._download_weights()


		self.save_dir = r"C:\Users\ASUS\Desktop\github_projects\face_attack_det\weights\paper"
		# self.checkpoint_path_g    = self.save_dir+"/gen/cp-{epoch:04d}.ckpt"
		self.checkpoint_path_g = os.path.join(self.save_dir, f"cp-0049.ckpt")
		# self.checkpoint_path_re   = self.save_dir+"/ReE/cp-{epoch:04d}.ckpt"
		# self.checkpoint_path_d1   = self.save_dir+"/dis1/cp-{epoch:04d}.ckpt"
		# self.checkpoint_path_d2   = self.save_dir+"/dis2/cp-{epoch:04d}.ckpt"
		# self.checkpoint_path_d3   = self.save_dir+"/dis3/cp-{epoch:04d}.ckpt"
		# self.checkpoint_path_g_op = self.save_dir+"/g_opt/cp-{epoch:04d}.ckpt"

		self.checkpoint_dir_g    = os.path.dirname(self.checkpoint_path_g)
		# self.checkpoint_dir_re   = os.path.dirname(self.checkpoint_path_re)
		# self.checkpoint_dir_d1   = os.path.dirname(self.checkpoint_path_d1)
		# self.checkpoint_dir_d2   = os.path.dirname(self.checkpoint_path_d2)
		# self.checkpoint_dir_d3   = os.path.dirname(self.checkpoint_path_d3)
		# self.checkpoint_dir_g_op = os.path.dirname(self.checkpoint_path_g_op)

		self.model_list  = [self.gen]#, self.RE, self.disc1, self.disc2, self.disc3]
		self.model_p_list= [self.checkpoint_path_g,
							# self.checkpoint_path_re, 
							# self.checkpoint_path_d1,
							# self.checkpoint_path_d2,
							# self.checkpoint_path_d3
							]
		self.model_d_list= [self.checkpoint_dir_g,
							# self.checkpoint_dir_re,
							# self.checkpoint_dir_d1,
							# self.checkpoint_dir_d2,
							# self.checkpoint_dir_d3
							]

		print("Loading models ...")
		for model_, model_dir_ in zip(self.model_list, self.model_d_list):
			current_checkpoint = os.path.join(model_dir_ , "cp-0049.ckpt")
			print("current_checkpoint", current_checkpoint)
			model_.load_weights(current_checkpoint).expect_partial()
		
		print("Models are loaded successfully")
	

	@staticmethod
	def _unzip(zip_file_path, extract_to):
		with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
			zip_ref.extractall(extract_to)

	def _download_weights(self):
		if self._attack_name == "paper":
			os.makedirs(os.path.join(os.getcwd(), "weights", "paper"), exist_ok=True)
			
			if not(os.path.isfile(os.path.join(os.getcwd(), "weights", "paper", "cp-0049.ckpt.index"))):
				where_to_download_zip_file = os.path.join(os.getcwd(), "weights", "paper", "cp-0049.ckpt.index")
				gdown.download(
					f"https://drive.google.com/uc?id={ckpt_configs.PAPER['INDEX']}",
					where_to_download_zip_file,
					quiet=False
				)
			if not(os.path.isfile(os.path.join(os.getcwd(), "weights", "paper", "cp-0049.ckpt.data-00000-of-00001"))):
				where_to_download_zip_file = os.path.join(os.getcwd(), "weights", "paper", "cp-0049.ckpt.data-00000-of-00001")
				gdown.download(
					f"https://drive.google.com/uc?id={ckpt_configs.PAPER['DATA']}",
					where_to_download_zip_file,
					quiet=False
				)
	


	@staticmethod
	def _normalization_score(score, shift=0.6, scale=1.6, lower=-0.4, upper=0.8):
		nor_score = (score+shift)/scale
		if nor_score < 0.32:
			if nor_score < 0:
				nor_score = 0
				return nor_score, 'Live' 
		else:
			if nor_score > 1:
				nor_score = 1
				return nor_score, 'Spoof'

	# @tf.function
	def _test_graph(self, img):

		dmap_pred, p, c, n, x, region_map = self.gen(img, training=False)
		dmap_score = tf.reduce_mean(dmap_pred[:,:,:,1], axis=[1,2]) - tf.reduce_mean(dmap_pred[:,:,:,0], axis=[1,2])
		p = tf.reduce_mean(p, axis=[1,2,3])

		final_score = dmap_score[0] + 0.1*p[0]
		final_score, decision = self._normalization_score(final_score)

		return decision, final_score
	
	def detect(self, img):
		res = self._face_det_model.detect(img)
		face = res["face"]
		face = cv2.resize(face, (256, 256))
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		# img, lm = self.image_process(img)
		# img = img / 255.

		cv2.imwrite("KKK_mine.png", face)
		img = face / 255.
		# img, lm, face = face_crop_and_resize(img, lm, 256, aug=False)
		
		img = img[np.newaxis, :, :, :]
		img = tf.convert_to_tensor(img)
		decision, conf = self._test_graph(img)

		print("decision", decision)
		if decision == "Spoof":
			return True, conf
		else:
			return False, conf