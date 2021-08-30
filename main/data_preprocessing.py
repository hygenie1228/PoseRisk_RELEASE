import os
import os.path as osp
import glob
import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import __init_path
from demo_dataset import CropDataset
from multi_person_tracker import MPT
from multi_person_tracker.data import video_to_images

MIN_SEC = 8
NUM_FRAMES = 200
BBOX_SCALE = 1.2

def main(src_dir):    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    mot = MPT(
                device=device,
                batch_size=6,
                display=False,
                detector_type='yolo',
                output_format='dict',
                yolo_img_size=416,
            )

    dir_names = sorted(glob.glob(osp.join(src_dir, '*')))
    # dir_names = sorted(glob.glob(osp.join(data_dir,video_code+'*')))

    for src_name in tqdm(dir_names[4:]):
        img_dir = src_name.replace('videos','images')
        os.makedirs(img_dir, exist_ok=True)
        processed_dir = src_name.replace('videos','processed_videos')
        os.makedirs(processed_dir, exist_ok=True)

        file_names = glob.glob(osp.join(src_name, '*')) + glob.glob(osp.join(src_name, '**', '*'))
        file_names = sorted(file_names)
        for file_name in tqdm(file_names):
            save_dir = file_name.split('/')[-1].split('.')[0]
            cap = cv2.VideoCapture(file_name)
            fps = cap.get(cv2.CAP_PROP_FPS)

            os.makedirs(osp.join(img_dir, save_dir), exist_ok=True)
            os.makedirs(osp.join(img_dir, save_dir, 'tmp'), exist_ok=True)
            print(osp.join(img_dir, save_dir))
            idx = 0
            while(cap.isOpened()):
                ret, frame = cap.read()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if ret == False:
                    break

                cv2.imwrite(osp.join(img_dir, save_dir, 'tmp', '{0:09d}.jpg'.format(idx)),frame)
                idx += 1

            cap.release()
            cv2.destroyAllWindows()
            del cap

            image_folder = osp.join(img_dir, save_dir, 'tmp')
            
            tracking_results = mot(image_folder) 
            filtered_results = []
            ####
            NUM_FRAMES = int(MIN_SEC * fps)
            
            for person_id in list(tracking_results.keys()):
                if tracking_results[person_id]['frames'].shape[0] >= NUM_FRAMES:
                    filtered_results.append(tracking_results[person_id])
            
            del tracking_results
            
            ### divide
            tracking_results = []
            for person_id, _ in tqdm(enumerate(filtered_results)):
                frame_len = len(filtered_results[person_id]['frames'])
                video_batch = frame_len // NUM_FRAMES

                for batch_i in range(video_batch):
                    results = {}
                    results['bbox'] = filtered_results[person_id]['bbox'][NUM_FRAMES*batch_i:NUM_FRAMES*(batch_i+1)]
                    results['frames'] = filtered_results[person_id]['frames'][NUM_FRAMES*batch_i:NUM_FRAMES*(batch_i+1)]
                    tracking_results.append(results)

            for person_id, _ in tqdm(enumerate(tracking_results)):
                bboxes = joints2d = None
                bboxes = tracking_results[person_id]['bbox']
                frames = tracking_results[person_id]['frames']

                dataset = CropDataset(
                    image_folder=image_folder,
                    frames=frames,
                    bboxes=bboxes,
                    joints2d=joints2d,
                    scale=BBOX_SCALE,
                ) 

                crop_dataloader = DataLoader(dataset, batch_size=256, num_workers=8)

                images = []
                for batch in crop_dataloader:
                    images.append(batch)

                images = torch.cat(images, dim=0)
                images = images.permute(0,2,3,1).cpu().numpy()
                images = images[:,:,:,::-1]*255
                batch_size, img_height, img_width, _ = images.shape

                save_img_path = osp.join(img_dir, save_dir, str(person_id))
                save_video_path = osp.join(processed_dir, f'{save_dir}_{str(person_id)}.mp4')
                
                os.makedirs(save_img_path, exist_ok=True)
                video_writer = cv2.VideoWriter(save_video_path, 0x7634706d, fps, (img_width, img_height))
                
                for frame_id in range(batch_size):
                    cv2.imwrite(osp.join(save_img_path, '{0:06d}.jpg'.format(frame_id)), images[frame_id])
                    video_writer.write(np.uint8(images[frame_id]))
                    
                video_writer.release()

            tmp_dir = osp.join(img_dir, save_dir, 'tmp')
            os.system(f'rm -rf {tmp_dir}')

if __name__ == "__main__":
    SRC_DIR = './data/NRF/videos/train'
    main(SRC_DIR)