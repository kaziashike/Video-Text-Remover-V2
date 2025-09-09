import copy
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor

# Define image preprocessing methods
_to_tensors = transforms.Compose([
    Stack(),  # Stack images as sequences
    ToTorchFormatTensor()  # Convert stacked images to PyTorch tensors
])


class STTNInpaint:
    def __init__(self):
        self.device = config.device
        # 1. Create an InpaintGenerator model instance and load it onto the selected device
        self.model = InpaintGenerator().to(self.device)
        # 2. Load the pre-trained model weights, load the model's state dictionary
        self.model.load_state_dict(torch.load(config.STTN_MODEL_PATH, map_location='cpu')['netG'])
        # 3. # Set the model to evaluation mode
        self.model.eval()
        # Model input width and height
        self.model_input_width, self.model_input_height = 640, 120
        # 2. Set adjacent frame count
        self.neighbor_stride = config.STTN_NEIGHBOR_STRIDE
        self.ref_length = config.STTN_REFERENCE_LENGTH

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: Original video frames
        :param mask: Subtitle area mask
        """
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # Determine the vertical height part to remove subtitles
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)
        # Initialize frame storage variables
        # High resolution frame storage list
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}  # Dictionary to store scaled frames
        comps = {}  # Dictionary to store completed frames
        # Store final video frames
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # Initialize a list for each removal part

        # Read and scale frames
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # Crop and scale for each removal part
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # Crop
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # Scale
                frames_scaled[k].append(image_resize)  # Add the scaled frame to the corresponding list

        # Process each removal part
        for k in range(len(inpaint_area)):
            # Call the inpaint function for processing
            comps[k] = self.inpaint(frames_scaled[k])

        # If there are removal parts
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # Take out the original frame
                # For each segment in the mode
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(comps[k][j], (W_ori, split_h))  # Scale the completed frame back to original size
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)  # Convert color space
                    # Get the mask area and perform image synthesis
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]  # Take out the mask area
                    # Implement image blending within the mask area
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                # Add the final frame to the list
                inpainted_frames.append(frame)
                print(f'processing frame, {len(frames_hr) - j} left')
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # Convert to binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        Sample reference frames from the entire video
        """
        # Initialize the reference frame index list
        ref_index = []
        # Iterate within the video length range step by step according to ref_length
        for i in range(0, length, self.ref_length):
            # If the current frame is not in the neighboring frames
            if i not in neighbor_ids:
                # Add it to the reference frame list
                ref_index.append(i)
        # Return the reference frame index list
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        """
        Use STTN to complete hole filling (hole is the masked area)
        """
        frame_length = len(frames)
        # Preprocess the frames to convert to tensors and normalize
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        # Transfer the feature tensor to the specified device (CPU or GPU)
        feats = feats.to(self.device)
        # Initialize a list of the same length as the video to store the processed frames
        comp_frames = [None] * frame_length
        # Disable gradient calculation, used in inference stage to save memory and accelerate
        with torch.no_grad():
            # Pass the processed frames through the encoder to generate feature representations
            feats = self.model.encoder(feats.view(frame_length, 3, self.model_input_height, self.model_input_width))
            # Get feature dimension information
            _, c, feat_h, feat_w = feats.size()
            # Adjust the feature shape to match the model's expected input
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        # Get the inpainting area
        # Loop through the video within the set neighbor frame stride
        for f in range(0, frame_length, self.neighbor_stride):
            # Calculate neighbor frame IDs
            neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            # Get reference frame indices
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            # Also disable gradient calculation
            with torch.no_grad():
                # Infer features through the model and pass them to the decoder to generate completed frames
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                # Pass the predicted features through the decoder to generate images, apply the tanh activation function, and then detach the tensor
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                # Rescale the result tensor to the range of 0 to 255 (image pixel values)
                pred_img = (pred_img + 1) / 2
                # Move the tensor back to CPU and convert to NumPy array
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # Traverse neighboring frames
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    # Convert the predicted image to unsigned 8-bit integer format
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        # If this position is empty, assign it to the newly calculated image
                        comp_frames[idx] = img
                    else:
                        # If there was already an image at this position, mix the new and old images to improve quality
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # Return the processed frame sequence
        return comp_frames

    @staticmethod
    def get_inpaint_area_by_mask(H, h, mask):
        """
        Get the subtitle removal area, determine the area and height to be filled based on the mask
        """
        # List to store the painting area
        inpaint_area = []
        # Start from the subtitle position at the bottom of the video, assuming subtitles are usually located at the bottom
        to_H = from_H = H
        # Traverse the mask from bottom to top
        while from_H != 0:
            if to_H - h < 0:
                # If the next segment will exceed the top, start from the top
                from_H = 0
                to_H = h
            else:
                # Determine the upper boundary of the segment
                from_H = to_H - h
            # Check if the current segment contains mask pixels
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                # If it's not the first segment, move down to ensure no mask area is missed
                if to_H != H:
                    move = 0
                    while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                        move += 1
                    # Ensure not to cross the bottom
                    if to_H + move < H and move < h:
                        to_H += move
                        from_H += move
                # Add the segment to the list
                if (from_H, to_H) not in inpaint_area:
                    inpaint_area.append((from_H, to_H))
                else:
                    break
            # Move to the next segment
            to_H -= h
        return inpaint_area  # Return the painting area list

    @staticmethod
    def get_inpaint_area_by_selection(input_sub_area, mask):
        print('use selection area for inpainting')
        height, width = mask.shape[:2]
        ymin, ymax, _, _ = input_sub_area
        interval_size = 135
        # List to store the result
        inpaint_area = []
        # Calculate and store standard intervals
        for i in range(ymin, ymax, interval_size):
            inpaint_area.append((i, i + interval_size))
        # Check if the last interval has reached the maximum value
        if inpaint_area[-1][1] != ymax:
            # If not, create a new interval starting from the end of the last interval and ending at the expanded value
            if inpaint_area[-1][1] + interval_size <= height:
                inpaint_area.append((inpaint_area[-1][1], inpaint_area[-1][1] + interval_size))
        return inpaint_area  # Return the painting area list


class STTNVideoInpaint:

    def read_frame_info_from_video(self):
        # Use opencv to read the video
        reader = cv2.VideoCapture(self.video_path)
        # Get the video's width, height, fps, and frame count information and store it in the frame_info dictionary
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),  # Video's original width
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),  # Video's original height
            'fps': reader.get(cv2.CAP_PROP_FPS),  # Video's fps
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # Video's total frame count
        }
        # Return the video reading object, frame information, and video writing object
        return reader, frame_info

    def __init__(self, video_path, mask_path=None, clip_gap=None):
        # STTNInpaint video repair instance initialization
        self.sttn_inpaint = STTNInpaint()
        # Video and mask paths
        self.video_path = video_path
        self.mask_path = mask_path
        # Set the path for the output video file
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        # Configure the maximum number of frames that can be loaded in one processing
        if clip_gap is None:
            self.clip_gap = config.STTN_MAX_LOAD_NUM
        else:
            self.clip_gap = clip_gap

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        reader = None
        writer = None
        try:
            # Read video frame information
            reader, frame_info = self.read_frame_info_from_video()
            if input_sub_remover is not None:
                writer = input_sub_remover.video_writer
            else:
                # Create video writing object, used to output the repaired video
                writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
            
            # Calculate the number of times the video needs to be iteratively repaired
            rec_time = frame_info['len'] // self.clip_gap if frame_info['len'] % self.clip_gap == 0 else frame_info['len'] // self.clip_gap + 1
            # Calculate the split height, used to determine the size of the repair area
            split_h = int(frame_info['W_ori'] * 3 / 16)
            
            if input_mask is None:
                # Read mask
                mask = self.sttn_inpaint.read_mask(self.mask_path)
            else:
                _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
                mask = mask[:, :, None]
                
            # Get the repair area position
            inpaint_area = self.sttn_inpaint.get_inpaint_area_by_mask(frame_info['H_ori'], split_h, mask)
            
            # Traverse each iteration time
            for i in range(rec_time):
                start_f = i * self.clip_gap  # Start frame position
                end_f = min((i + 1) * self.clip_gap, frame_info['len'])  # End frame position
                print('Processing:', start_f + 1, '-', end_f, ' / Total:', frame_info['len'])
                
                frames_hr = []  # High resolution frame list
                frames = {}  # Frame dictionary, used to store cropped images
                comps = {}  # Combination dictionary, used to store repaired images
                
                # Initialize frame dictionary
                for k in range(len(inpaint_area)):
                    frames[k] = []
                    
                # Read and repair high resolution frames
                valid_frames_count = 0
                for j in range(start_f, end_f):
                    success, image = reader.read()
                    if not success:
                        print(f"Warning: Failed to read frame {j}.")
                        break
                    
                    frames_hr.append(image)
                    valid_frames_count += 1
                    
                    for k in range(len(inpaint_area)):
                        # Crop, scale, and add to frame dictionary
                        image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                        image_resize = cv2.resize(image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                        frames[k].append(image_resize)
                
                # If no valid frames were read, skip the current iteration
                if valid_frames_count == 0:
                    print(f"Warning: No valid frames found in range {start_f+1}-{end_f}. Skipping this segment.")
                    continue
                    
                # Run repair for each repair area
                for k in range(len(inpaint_area)):
                    if len(frames[k]) > 0:  # Ensure there are frames to process
                        comps[k] = self.sttn_inpaint.inpaint(frames[k])
                    else:
                        comps[k] = []
                
                # If there are areas to repair
                if inpaint_area and valid_frames_count > 0:
                    for j in range(valid_frames_count):
                        if input_sub_remover is not None and input_sub_remover.gui_mode:
                            original_frame = copy.deepcopy(frames_hr[j])
                        else:
                            original_frame = None
                            
                        frame = frames_hr[j]
                        
                        for k in range(len(inpaint_area)):
                            if j < len(comps[k]):  # Ensure index is valid
                                # Resize the repaired image back to original resolution and blend it into the original frame
                                comp = cv2.resize(comps[k][j], (frame_info['W_ori'], split_h))
                                comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                                mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                                frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                        
                        writer.write(frame)
                        
                        if input_sub_remover is not None:
                            if tbar is not None:
                                input_sub_remover.update_progress(tbar, increment=1)
                            if original_frame is not None and input_sub_remover.gui_mode:
                                input_sub_remover.preview_frame = cv2.hconcat([original_frame, frame])
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            # Do not raise an exception, allowing the program to continue executing
        finally:
            if writer:
                writer.release()


if __name__ == '__main__':
    mask_path = '../../test/test.png'
    video_path = '../../test/test.mp4'
    # Record start time
    start = time.time()
    sttn_video_inpaint = STTNVideoInpaint(video_path, mask_path, clip_gap=config.STTN_MAX_LOAD_NUM)
    sttn_video_inpaint()
    print(f'video generated at {sttn_video_inpaint.video_out_path}')
    print(f'time cost: {time.time() - start}')
