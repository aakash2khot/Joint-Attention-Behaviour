def score1(p1, dp1, p2, dp2):
    sx=dp2[0]*dp1[0]+dp2[1]*dp1[1]
    sy=dp2[0]*(-dp1[1])+dp2[1]*(dp1[0])
    if sy==0:
        return None
    PQx = p2[0] - p1[0]
    PQy = p2[1] - p1[1]
    rx = dp1[0]
    ry = dp1[1]
    rxt = -ry
    ryt = rx
    qx = PQx * rx + PQy * ry
    qy = PQx * rxt + PQy * ryt
    a = qx - qy * sx / sy
    return (p1[0]+a*rx, p1[1]+a*ry)

print(score1((0,0),(1,0),(5,6),(0,-1)))