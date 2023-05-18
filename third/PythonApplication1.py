import os
os.add_dll_directory(r"C:\Users\kopyl\opencv\build2\install\x64\vc16\bin")
os.add_dll_directory(r"C:\Users\kopyl\source\repos\Dll1\x64\Release")
import opencv_job
import k4a
import matplotlib.pyplot as plt
import cv2
import numpy as np
import win32api, win32con
import time
import itertools


def click(x,y, thats):
    win32api.SetCursorPos((x,y))
    if thats:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)


keep_ploting = True


def on_key(event):
    global keep_ploting
    keep_ploting = False

if __name__ == '__main__':
    # Open a device using the static function Device.open().
    device = k4a.Device.open()
    device_config = k4a.DeviceConfiguration(
        color_format=k4a.EImageFormat.COLOR_BGRA32,
        color_resolution=k4a.EColorResolution.RES_1080P,
        depth_mode=k4a.EDepthMode.NFOV_UNBINNED,
        camera_fps=k4a.EFramesPerSecond.FPS_15,
        synchronized_images_only=True,
        depth_delay_off_color_usec=0,
        wired_sync_mode=k4a.EWiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec=0,
        disable_streaming_indicator=False)
    device.start_cameras(device_config)
    fig = plt.figure()
    fig.show()
    ax = []
    ax.append(fig.add_subplot(1, 1, 1, label="IR"))
    im = []
    capture = device.get_capture(-1)
    im.append(ax[0].imshow(capture.ir.data, cmap='jet'))
    q = capture.ir.data
    q = list(itertools.chain.from_iterable(q))
    ax[0].title.set_text('Color')
    x = 0
    y = 0
    prev_x = x
    prev_y = y
    prev = q
    cnt_frame = 0
    www = 1
    tex = plt.text(0, 0, str(1), fontsize=12, color = 'g')
    cords = [118, 188, 433, 342]
    circle1 = plt.Circle((cords[0], cords[1]), 10, color='r')
    circle2 = plt.Circle((cords[2], cords[3]), 10, color='r')
    circle3 = plt.Circle((int(prev_x + cords[0]),int(prev_y+cords[1])), 10, color='b')
    ax[0].add_patch(circle1)
    ax[0].add_patch(circle2)
    ax[0].add_patch(circle3)
    while keep_ploting:
        start = time.time_ns()
        plt.pause(0.01)
        plt.draw()
        capture = device.get_capture(-1)
        if capture is None:
            del fig
            break
        q = capture.ir.data
        im[0].set_data((q))
        #q = list(q.flatten())
        q = list(itertools.chain.from_iterable(q))
        ans = np.array([0, 0], dtype=np.uint8)
        if(type(prev) != list):
            prev = list(prev.flatten())
        #clib.loading(q.astype(np.uint8), prev.astype(np.uint8), cords.astype(np.uint8), ans.astype(np.uint8))
        s = opencv_job.interface(q, prev, cords, [576], [640])
        prev = q
        end = 1/((time.time_ns() - start)/1e9)
        tex.set_text(str(end))
        if(cnt_frame == 0 and s[0] != prev_x and s[1] != prev_y and s[0] != cords[0] and s[1] != cords[1]):
            click(int((s[0]+cords[0])/567*1920),int(1080 - (s[1]+cords[1])/640*1080), False)
            cnt_frame+=1
            circle3.center = s[0]+cords[0], s[1]+cords[1]
            prev_x = s[0]
            prev_y = s[1]
        else:
            
            if(cnt_frame != 0):
                if cnt_frame == www:
                    cnt_frame = 0
                else:
                    cnt_frame+=1

        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # There is no need to delete the capture since Python will take care of
        # that in the object's deleter.


    # There is no need to stop the cameras since the deleter will stop
    # the cameras, but it's still prudent to do it explicitly.
    device.stop_cameras()
