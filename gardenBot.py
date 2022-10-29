#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import argparse
import time
from datetime import datetime
from xml.dom import minidom
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

rtmp_url = "rtmp://recipe.annoyingprojects.com/live/farmbot"


def current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time)
    return str(current_time)
    
def current_date_time():
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    print(current_time)
    return str(current_time)
    
def daily_image_time():
    file = minidom.parse('config/config.xml')
    Daily_Time_Image = file.getElementsByTagName('Daily_Time_Image')
    return Daily_Time_Image
    


hostName = "localhost"
serverPort = 8080

class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>https://pythonbasics.org</title></head>", "utf-8"))
        self.wfile.write(bytes("<p>Request: %s</p>" % self.path, "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This is an example web server.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))

if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))
    parser = argparse.ArgumentParser()
parser.add_argument(
    "-res",
    "--resolution",
    type=str,
    default="720",
    help="Sets the resolution on mono cameras. Options: 800 | 720 | 400",
)
parser.add_argument(
    "-md",
    "--mesh_dir",
    type=str,
    default=None,
    help="Output directory for mesh files. If not specified mesh files won't be saved",
)
parser.add_argument(
    "-lm",
    "--load_mesh",
    default=False,
    action="store_true",
    help="Read camera intrinsics, generate mesh files and load them into the stereo node.",
)
parser.add_argument(
    "-rect",
    "--out_rectified",
    default=False,
    action="store_true",
    help="Generate and display rectified streams",
)
parser.add_argument(
    "-lr",
    "--lrcheck",
    default=False,
    action="store_true",
    help="Better handling for occlusions",
)
parser.add_argument(
    "-e",
    "--extended",
    default=False,
    action="store_true",
    help="Closer-in minimum depth, disparity range is doubled",
)
parser.add_argument(
    "-s",
    "--subpixel",
    default=False,
    action="store_true",
    help="Better accuracy for longer distance, fractional disparity 32-levels",
)
parser.add_argument(
    "-m",
    "--median",
    type=str,
    default="7x7",
    help="Choose the size of median filtering. Options: OFF | 3x3 | 5x5 | 7x7 (default)",
)
parser.add_argument(
    "-d",
    "--depth",
    default=False,
    action="store_true",
    help="Display depth frames",
)
args = parser.parse_args()

resolutionMap = {"800": (1280, 800), "720": (1280, 720), "400": (640, 400)}
if args.resolution not in resolutionMap:
    exit("Unsupported resolution!")

resolution = resolutionMap[args.resolution]
meshDirectory = args.mesh_dir  # Output dir for mesh files
generateMesh = args.load_mesh  # Load mesh files

outRectified = args.out_rectified  # Output and display rectified streams
lrcheck = args.lrcheck  # Better handling for occlusions
extended = args.extended  # Closer-in minimum depth, disparity range is doubled
subpixel = args.subpixel  # Better accuracy for longer distance, fractional disparity 32-levels
depth = args.depth  # Display depth frames

medianMap = {
    "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
    "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
    "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
    "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
}
if args.median not in medianMap:
    exit("Unsupported median size!")

median = medianMap[args.median]

print("StereoDepth config options:")
print("    Resolution:  ", resolution)

print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)
print("    Generating mesh files:  ", generateMesh)
print("    Outputting mesh files to:  ", meshDirectory)


def getMesh(calibData):
    M1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, resolution[0], resolution[1]))
    d1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
    R1 = np.array(calibData.getStereoLeftRectificationRotation())
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, resolution[0], resolution[1]))
    d2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
    R2 = np.array(calibData.getStereoRightRectificationRotation())
    mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, M2, resolution, cv2.CV_32FC1)
    mapXR, mapYR = cv2.initUndistortRectifyMap(M2, d2, R2, M2, resolution, cv2.CV_32FC1)

    meshCellSize = 16
    meshLeft = []
    meshRight = []

    for y in range(mapXL.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            rowRight = []
            for x in range(mapXL.shape[1] + 1):
                if x % meshCellSize == 0:
                    if y == mapXL.shape[0] and x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y - 1, x - 1])
                        rowLeft.append(mapXL[y - 1, x - 1])
                        rowRight.append(mapYR[y - 1, x - 1])
                        rowRight.append(mapXR[y - 1, x - 1])
                    elif y == mapXL.shape[0]:
                        rowLeft.append(mapYL[y - 1, x])
                        rowLeft.append(mapXL[y - 1, x])
                        rowRight.append(mapYR[y - 1, x])
                        rowRight.append(mapXR[y - 1, x])
                    elif x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y, x - 1])
                        rowLeft.append(mapXL[y, x - 1])
                        rowRight.append(mapYR[y, x - 1])
                        rowRight.append(mapXR[y, x - 1])
                    else:
                        rowLeft.append(mapYL[y, x])
                        rowLeft.append(mapXL[y, x])
                        rowRight.append(mapYR[y, x])
                        rowRight.append(mapXR[y, x])
            if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
                rowRight.append(0)
                rowRight.append(0)

            meshLeft.append(rowLeft)
            meshRight.append(rowRight)

    meshLeft = np.array(meshLeft)
    meshRight = np.array(meshRight)

    return meshLeft, meshRight


def saveMeshFiles(meshLeft, meshRight, outputPath):
    print("Saving mesh to:", outputPath)
    meshLeft.tofile(outputPath + "/left_mesh.calib")
    meshRight.tofile(outputPath + "/right_mesh.calib")


def getDisparityFrame(frame):
    maxDisp = stereo.initialConfig.getMaxDisparity()
    disp = (frame * (255.0 / maxDisp)).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    return disp


print("Creating Stereo Depth pipeline")
pipeline = dai.Pipeline()

camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutDisparity = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutRectifLeft = pipeline.create(dai.node.XLinkOut)
xoutRectifRight = pipeline.create(dai.node.XLinkOut)

camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(1920, 1080)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Linking
camRgb.video.link(xoutVideo.input)


command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(1920, 1080),
           '-r', str(30),
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmp_url]
   
res = (
    dai.MonoCameraProperties.SensorResolution.THE_800_P
    if resolution[1] == 800
    else dai.MonoCameraProperties.SensorResolution.THE_720_P
    if resolution[1] == 720
    else dai.MonoCameraProperties.SensorResolution.THE_400_P
)
for monoCam in (camLeft, camRight):  # Common config
    monoCam.setResolution(res)
    # monoCam.setFps(20.0)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)  # KERNEL_7x7 default
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)

xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")
xoutDisparity.setStreamName("disparity")
xoutVideo.setStreamName("video")
xoutDepth.setStreamName("depth")
xoutRectifLeft.setStreamName("rectifiedLeft")
xoutRectifRight.setStreamName("rectifiedRight")

camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)
stereo.syncedLeft.link(xoutLeft.input)
stereo.syncedRight.link(xoutRight.input)
stereo.disparity.link(xoutDisparity.input)
if depth:
    stereo.depth.link(xoutDepth.input)
if outRectified:
    stereo.rectifiedLeft.link(xoutRectifLeft.input)
    stereo.rectifiedRight.link(xoutRectifRight.input)

streams = ["left", "right", "video"]
if outRectified:
    streams.extend(["rectifiedLeft", "rectifiedRight"])
streams.append("disparity")
if depth:
    streams.append("depth")

calibData = dai.Device().readCalibration()
leftMesh, rightMesh = getMesh(calibData)
if generateMesh:
    meshLeft = list(leftMesh.tobytes())
    meshRight = list(rightMesh.tobytes())
    stereo.loadMeshData(meshLeft, meshRight)

if meshDirectory is not None:
    saveMeshFiles(leftMesh, rightMesh, meshDirectory)


print("Creating DepthAI device")
#p = subprocess.Popen(command, stdin=subprocess.PIPE)
with dai.Device(pipeline) as device:
    # Create a receive queue for each stream
    qList = [device.getOutputQueue(stream, 8, blocking=False) for stream in streams]
    timeArray = daily_image_time()
    while True:
       
          for q in qList:
            name = q.getName()
            frame = q.get().getCvFrame()
            if name == "depth":
                frame = frame.astype(np.uint16)
            elif name == "disparity":
                frame = getDisparityFrame(frame)
            elif name == "video":
                print("TEst")
                #p.stdin.write(frame.tobytes())
            for time in timeArray:     
                if current_time() == time.firstChild.data:    
                   fileName = current_date_time()+"-"+q.getName()+".jpg"
                   cv2.imwrite(filename='daily_images/'+fileName, img=frame)  
            cv2.imshow(name, frame)    

          if cv2.waitKey(1) == ord("q"):
            break

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

