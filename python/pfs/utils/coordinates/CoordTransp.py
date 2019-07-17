#!/usr/bin/env python

import os,sys,re
import math as mt
import numpy as np
import scipy
from scipy import interpolate as ipol

## Dictionary keys ( argument name is mode)
# sky_pfi : sky to F3C
# sky_pfi_hsc : sky to hsc focal plane
# sky_mcs : sky to MCS
# pfi_mcs : F3C to MCS
# pfi_mcs_wofe : F3C to MCS (w/o Field Element)
# mcs_pfi : MCS to F3C

import DistortionCoefficients as DCoeff

mypath=os.path.dirname(os.path.abspath(__file__))+'/'

## input x,y point list, zenith angle, mode, rotator angle, centerposition
def CoordinateTransform(xyin, za, mode, inr=0., cent=np.array([[0.],[0.]])):
    
    #print(mode,file=sys.stderr)
    c=DCoeff.Coeff(mode)

    #convert pixel to mm: mcs_pfi
    if c.mode=='mcs_pfi' :
        xyin=Pixel2mm(xyin, inr, cent)

    arg=np.array([np.arctan2(j,i) for i,j in zip(*xyin)])
    # MCS and PFI : xy is flipped
    if c.mode == 'mcs_pfi' :
        arg=arg+np.pi

    # PFI to MCS: input argument depends on rotator angle
    if c.mode == 'pfi_mcs'  or c.mode == 'pfi_mcs_wofe':
        arg=arg+np.deg2rad(inr)+np.pi


    # Scale conversion
    print("Scaling", file=sys.stderr)
    scale=ScalingFactor(xyin, c)

    #deviation
    # base
    """
    To check the distortion, the below forking is commented out temporary
    """
    """
    if mode == 'mcs_pfi' :
        print("Offset 1 is skipped", file=sys.stderr)
        offx1=np.zeros(xyin.shape[1])
        #print(offx1)
        offy1=np.zeros(xyin.shape[1])
    else :
        print("Offset 1", file=sys.stderr)
        offx1,offy1=OffsetBase(xyin, c)
    """

    if c.mode=='pfi_mcs' or c.mode == 'pfi_mcs_wofe' :
        offx1=np.zeros(xyin.shape[1]) ; offy1=np.zeros(xyin.shape[1])
        offx2=np.zeros(xyin.shape[1]) ; offy2=np.zeros(xyin.shape[1])
    else:
        print("Offset 1", file=sys.stderr)
        offx1,offy1=OffsetBase(xyin, c)

        # z-dependent
        print("Offset 2", file=sys.stderr)
        offx2,offy2=DeviationZenithAngle(xyin, za, c)
        #print(offx2)

    xx=scale*np.cos(arg)+offx1+offx2
    yy=scale*np.sin(arg)+offy1+offy2
    #print(xx)

    #print zip(scale,arg,offx1,offy1,offx2,offy2)
    #for s,t,ox1,oy1,ox2,oy2 in zip(scale,arg,offx1,offy1,offx2,offy2):
    #    x=s*mt.cos(t)+ox1+ox2
    #    y=s*mt.sin(t)+oy1+oy2
    #    #print x,y,x+y
    #    #xyout.append([x,y])
    #    xyout=np.append(xyout,[x,y,s,t,ox1,oy1,ox2,oy2], axis=0)

    #convert pixel to mm: mcs_pfi
    if c.mode=='pfi_mcs' or c.mode == 'pfi_mcs_wofe':
        xx,yy=mm2Pixel(xx, yy, cent)


    xyout=np.array([xx,yy,scale,arg,offx1,offy1,offx2,offy2])

    #print xyout
    return xyout

# differential : z
def DeviationZenithAngle(xyin, za, c):

    if c.mode == 'mcs_pfi':
        # Reference: zenith angle 60 deg
        coeffzx=DiffCoeff(za, c, axis='x')/DiffCoeff(60., c, axis='x')
        coeffzy=DiffCoeff(za, c, axis='y')/DiffCoeff(60., c, axis='y')
    else:
        # Reference: zenith angle 30 deg
        coeffz=DiffCoeff(za, c)/DiffCoeff(30., c)

    # x : dx = c0*x*y
    cx=c.cx
    # y : dy = c0*x^2 + c1*y^2 + c2*y^4 + c3*x^2*y^2 + c4
    cy=c.cy

    # y : slope cy5(z) * y
    za_a=[0.,30.,60.]
    sl_a=c.slp

    sl_itrp=ipol.splrep(za_a,sl_a,k=2,s=0)
    cy5=ipol.splev(za,sl_itrp)

    if c.mode == 'mcs_pfi':
        tarr=np.array([RotationPattern(za,c,x,y) for x,y in zip(*xyin)])
        rotxy=tarr.transpose()

        offx=np.array([coeffzx*(cx[0]*x*x+cx[1]*y*y+cx[2]*x*y+cx[3]) for x,y in zip(*rotxy)])
        offy=np.array([coeffzy*(cy[0]*x*x+cy[1]*y*y+cy[2]*np.power(y,4.)+cy[3]*x*x*y*y+cy[4])+cy5*y for x,y in zip(*rotxy)])
    else:
        offx=np.array([coeffz*cx*x*y for x,y in zip(*xyin)])
        offy=np.array([coeffz*(cy[0]*x*x+cy[1]*y*y+cy[2]*np.power(y,4.)+cy[3]*x*x*y*y+cy[4])+cy5*y for x,y in zip(*xyin)])

    #offx=[]
    #offy=[]
    #for x,y in zip(*xyin):
    #    #print x,y
    #    dx=coeffz*cx*x*y
    #    dy=coeffz*(cy[0]*x*x+cy[1]*y*y+cy[2]*np.power(y,4.)+cy[3]*x*x*y*y+cy[4])+cy5*y
    #    offx.extend([dx])
    #    offy.extend([dy])

    return offx,offy

def DiffCoeff(za, c, axis='x'):

    za*=np.pi/180.
    if (c.mode == 'mcs_pfi') and (axis == 'y'):
        return c.dsc[2]*(c.dsc[3]*np.sin(za)+(1-np.cos(za)))
    else:
        return c.dsc[0]*(c.dsc[1]*np.sin(za)+(1-np.cos(za)))

## treat rotated pattern (Sep. 2018)
def RotationPattern(za,c,x,y):
    
    ra=-1.*(60.-za)

    #if c.mode == 'mcs_pfi' :
    #    ra=ra+np.pi

    rx, ry = Rotation(x,y,ra)

    return rx, ry


## Offset at base
def OffsetBase(xyin, c):

    # sky-x sky-y off-x off-y
    dfile=mypath+"data/offset_base_"+c.mode+".dat"
    fi=open(dfile)
    line=fi.readlines()
    fi.close

    lines=[i.split() for i in line]
    #IpolD=map(list, zip(*lines))
    IpolD=np.swapaxes(np.array(lines,dtype=float),0,1)


    #print(IpolD)

    #x_itrp=ipol.bisplrep(IpolD[0,:],IpolD[1,:],IpolD[2,:],s=0)
    #y_itrp=ipol.bisplrep(IpolD[0,:],IpolD[1,:],IpolD[3,:],s=0)
    x_itrp=ipol.SmoothBivariateSpline(IpolD[0,:],IpolD[1,:],IpolD[2,:],kx=5,ky=5,s=1)
    y_itrp=ipol.SmoothBivariateSpline(IpolD[0,:],IpolD[1,:],IpolD[3,:],kx=5,ky=5,s=1)

    print("Interpol Done.", file=sys.stderr)

    #print(zip(*xyin))

    offsetx=np.array([x_itrp.ev(i,j) for i,j in zip(*xyin)])
    offsety=np.array([y_itrp.ev(i,j) for i,j in zip(*xyin)])

    #offsetx=[]
    #offsety=[]
    #for i,j in zip(*xyin):
    #    #print(i,j)
    #    offsetx.extend(x_itrp.ev([i],[j]))
    #    offsety.extend(y_itrp.ev([i],[j]))

    return offsetx,offsety

## Scaling Factor: function of r + interpol
def ScalingFactor(xyin, c):

    dist=[mt.sqrt(i*i+j*j) for i,j in zip(*xyin)]

    # scale1 : rfunction
    # scale2 : interpolation
    scale1=[ScalingFactor_Rfunc(r,c) for r in dist]
    if c.mode != 'pfi_mcs' and c.mode != 'pfi_mcs_wofe':
        # Derive Interpolation function
        sc_intf=ScalingFactor_Inter(c)
        scale2=ipol.splev(dist,sc_intf)

    if c.mode == 'pfi_mcs' or c.mode == 'pfi_mcs_wofe':
        scale=scale1
    else:
        scale=np.array([ x+y for x,y in zip(scale1, scale2)])

    return scale

## Scaling Factor: function of r
def ScalingFactor_Rfunc(r, c):

    rc=c.rsc

    if c.mode == 'mcs_pfi':
        return rc[0]*r+rc[1]*np.power(r,3.)+rc[2]*np.power(r,5.)+rc[3]*np.power(r,7.)
    else:
        return rc[0]*r+rc[1]*np.power(r,3.)+rc[2]*np.power(r,5.)

## Scaling Factor: interpolation func.
def ScalingFactor_Inter(c):

    dfile=mypath+"data/scale_interp_"+c.mode+".dat"
    fi=open(dfile)
    line=fi.readlines()
    fi.close

    lines=[i.split() for i in line]
    IpolD=list(map(list, zip(*lines)))

    r_itrp=ipol.splrep(IpolD[0],IpolD[1],s=0)

    return r_itrp

def Pixel2mm(xyin, inr, cent):


    offxy=xyin-cent

    xymm=[]
    for x,y in zip(*offxy):
        rx,ry = Rotation(x, y, inr)
        rx *= DCoeff.mcspixel ; ry *= DCoeff.mcspixel
        xymm.append([rx, -1*ry])

    xymm=np.swapaxes(np.array(xymm,dtype=float),0,1)

#    print(offxy,xymm)
    return xymm

def mm2Pixel(x, y, cent):

    sx = x/DCoeff.mcspixel + cent[0]
    sy = (-1.)*y/DCoeff.mcspixel + cent[1]

    return sx,sy

## Rotation (angle in degree)
def Rotation(x,y,rot):
    
    ra = np.deg2rad(rot)

    rx=np.cos(ra)*x-np.sin(ra)*y
    ry=np.sin(ra)*x+np.cos(ra)*y

    return rx, ry
