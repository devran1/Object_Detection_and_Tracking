#!/usr/bin/python3.8
import time
import numpy
import cv2
import sdl2
import sdl2.ext  # pip install pysdl2 download sdl2 website  and paste system32

class Display(object):
    def __init__(self, w, h):
        sdl2.ext.init()
        self.w, self.h = w, h
        self.Window = sdl2.ext.Window("slam", size=(w,h), position=(10,10)) #place on the screen of computer
        self.Window.show() #just shows it? opens it
        
    def paint(self, img):    
        #junk
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)
                
        # draw
        surf = sdl2.ext.pixels3d(self.Window.get_surface())
        
        surf[:, :, 0:3] = img.swapaxes(0,1) #3    #could not broadcast input array from shape (500,500,3) into shape (500,500,4) where is the other 3
        #blit
        self.Window.refresh()
