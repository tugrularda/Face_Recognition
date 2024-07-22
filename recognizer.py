import cv2
import os
import shutil
import networks
from datasetmanager import *
from test import *
import torchvision


# Device settings
if(torch.cuda.is_available()):
    device = 'cuda'
    print("\033[32mCUDA is used as the device.\033[0m")
else:
    device = 'cpu'
    print("\033[31mCPU is used as the device.\033[0m")


transform_ = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    

recognizer_dir = 'C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Data\\Recognizer'
database_dir = os.path.join(recognizer_dir, 'Database')
run_dir = os.path.join(recognizer_dir, 'Run')


#Model
model_path = 'C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Face_Recognition\\models\\siamese_model_f2.pth'
model = networks.SiameseNetwork(in_channels=3, in_size=256).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


cam = cv2.VideoCapture(0)
cv2.namedWindow("Capturing...")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Capturing...", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k%256 == 32:
        # SPACE pressed
        img_counter += 1
        img_name = f"opencv_frame_{img_counter}.png"
        img_path = os.path.join(run_dir, img_name)
        print(cv2.imwrite(img_name, frame))
        shutil.move(os.path.join('C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Face_Recognition', img_name), run_dir)
        print(f"{img_name} written!")

        # Data preparation
        transformed_img = transform_(read_image(img_path).float())
        database_data = TestImageDataset(database_dir)
        distances = []

        for db_img, db_label in database_data:
            db_img = db_img.to(device)
            transformed_img = transformed_img.to(device)

            dist = calculated_distance(model, transformed_img, db_img)
            print(dist)
            distances.append((dist, db_label))

        print(f"Predicted as {min(distances)}")

    elif k%256 == 78 or k%256 == 110:
        # n key pressed
        
        #img_counter += 1
        label = input('Enter your name: ')  
        img_name = f"opencv_frame_{label}.jpg"
        print(cv2.imwrite(img_name, frame))
        shutil.move(os.path.join('C:\\Users\\ardad\\Desktop\\Yazılım ve Elektronik\\Face_Recognition', img_name), database_dir)
        
        print(f"{img_name} written!") 

        with open(os.path.join(database_dir, 'labels.txt'), 'a') as file:
            file.write(f'{img_name}, {label} \n')

cam.release()

cv2.destroyAllWindows()