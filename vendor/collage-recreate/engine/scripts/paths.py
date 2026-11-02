"""Interpolate anchors once, then reuse the continuous dash stroker."""
import math


def smooth_points(points):
    if len(points)<3:return points
    out=[tuple(points[0])]
    for i in range(len(points)-1):
        p0=points[max(0,i-1)];p1=points[i];p2=points[i+1];p3=points[min(len(points)-1,i+2)]
        # Centripetal Catmull-Rom converted through recursive interpolation;
        # duplicate endpoint anchors are extrapolated to avoid zero knot spans.
        if p0==p1:p0=[2*p1[j]-p2[j] for j in range(2)]
        if p2==p3:p3=[2*p2[j]-p1[j] for j in range(2)]
        if p1==p2:continue
        ts=[0.]
        for a,b in zip([p0,p1,p2],[p1,p2,p3]):ts.append(ts[-1]+max(1e-6,math.dist(a,b)**.5))
        def mix(a,b,lo,hi,t):return [(hi-t)/(hi-lo)*a[j]+(t-lo)/(hi-lo)*b[j] for j in range(2)]
        steps=max(8,min(4096,math.ceil((math.dist(p0,p1)+math.dist(p1,p2)+math.dist(p2,p3))*2)))
        for k in range(1,steps+1):
            t=ts[1]+(ts[2]-ts[1])*k/steps
            a1=mix(p0,p1,ts[0],ts[1],t);a2=mix(p1,p2,ts[1],ts[2],t);a3=mix(p2,p3,ts[2],ts[3],t)
            b1=mix(a1,a2,ts[0],ts[2],t);b2=mix(a2,a3,ts[1],ts[3],t)
            out.append(tuple(mix(b1,b2,ts[1],ts[2],t)))
    return out
