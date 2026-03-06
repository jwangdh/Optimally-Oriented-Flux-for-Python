import numpy as np
from scipy import special as sp_special


# The method was first proposed by Max Law
# The python code is developed by Jierong WANG (jwangdh@connect.ust.hk)


def eigenvaluefield33(a11, a12, a13, a22, a23, a33):
    b = a11 + 1e-50
    d = a22 + 1e-50
    j = a33 + 1e-50

    c = -(a12 * a12 + a13 * a13 + a23 * a23 - b * d - d * j - j * b)
    d = -(b * d * j - a23 * a23 * b - a12 * a12 * j - a13 * a13 * d + 2 * a13 * a12 * a23)
    b = -a11 - a22 - a33 - 1e-50 - 1e-50 - 1e-50
    d = d + (2 * b * b * b - 9 * b * c) / 27

    c = (b * b) / 3 - c
    c = (c * c * c) / 27
    c = np.maximum(c, 0)
    c = np.sqrt(c)

    j = pow(c, 1 / 3)
    c = c + 1 * (c == 0)
    d = -d / 2 / c
    d = np.minimum(d, 1)
    d = np.maximum(d, -1)
    d = np.real(np.arccos(d) / 3)

    c = j * np.cos(d)
    d = j * np.sqrt(3) * np.sin(d)
    b = -b / 3

    j = -c - d + b
    d = -c + d + b
    b = 2 * c + b

    return b, j, d


# Given an eigenvalue v, compute the corresponding eigenvector
def eigenvectorfield33(a11, a12, a13, a22, a23, a33, v):
    # a11: a, a12: b, a13: c, a22: d, a23: e, a33: f
    # xy-plane implementation
    # vector = cross([a11-v;a12;a13], [a12;a22-v;a23])
    v1 = a12 * a23 - a13 * (a22 - v)
    v2 = a12 * a13 - a23 * (a11 - v)
    v3 = (a11 - v) * (a22 - v) - a12 * a12

    # xz-plane implementation
    # vector = cross([a11-v;a12;a13], [a13;a23;a33-v])
    v1_tmp = a12 * (a33 - v) - a13 * a23
    v2_tmp = a13 * a13 - (a11 - v) * (a33 - v)
    v3_tmp = a23 * (a11 - v) - a12 * a13

    condition = (np.abs(v1_tmp) + np.abs(v2_tmp) + np.abs(v3_tmp)) > (np.abs(v1) + np.abs(v2) + np.abs(v3))
    v1[condition] = v1_tmp[condition]
    v2[condition] = v2_tmp[condition]
    v3[condition] = v3_tmp[condition]

    # yz-plane implementation
    v1_tmp = (a22 - v) * (a33 - v) - a23 * a23
    v2_tmp = a23 * a13 - a12 * (a33 - v)
    v3_tmp = a12 * a23 - a13 * (a22 - v)

    condition = (np.abs(v1_tmp) + np.abs(v2_tmp) + np.abs(v3_tmp)) > (np.abs(v1) + np.abs(v2) + np.abs(v3))
    v1[condition] = v1_tmp[condition]
    v2[condition] = v2_tmp[condition]
    v3[condition] = v3_tmp[condition]

    condition = (np.abs(v1) + np.abs(v2) + np.abs(v3)) < 0.0000000001
    v1[condition & ((a11 - v) == 0)] = 1
    v2[condition & ((a22 - v) == 0)] = 1
    v3[condition & ((a33 - v) == 0)] = 1

    mag = np.sqrt(v1 * v1 + v2 * v2 + v3 * v3) + 1e-23
    v1 = v1 / mag
    v2 = v2 / mag
    v3 = v3 / mag

    return v1, v2, v3


def ifftshiftedcoormatrix(dimension):
    dim = dimension.shape[0]
    p = np.floor(dimension / 2)

    out = []

    for i in range(dim):
        a = (np.concatenate((np.arange(p[i] + 1, dimension[i] + 1), np.arange(1, p[i] + 1)))) - p[i] - 1
        reshapePara = np.ones((dim,), dtype="i")
        reshapePara[i] = dimension[i]
        A = a.reshape(reshapePara)
        repmatPara = dimension.copy()
        repmatPara[i] = 1
        out.append(np.tile(A, repmatPara))

    return out


def ifftshiftedcoordinate(dimension, dimIndex, pixelSpacing):
    dim = dimension.shape[0]
    p = np.floor(dimension / 2)

    a = (np.concatenate((np.arange(p[dimIndex] + 1, dimension[dimIndex] + 1), np.arange(1, p[dimIndex] + 1)))) - p[
        dimIndex] - 1
    a = a / pixelSpacing[dimIndex] / dimension[dimIndex]
    reshapePara = np.ones((dim,), dtype="i")
    reshapePara[dimIndex] = dimension[dimIndex]
    A = a.reshape(reshapePara)
    repmatPara = dimension.copy()
    repmatPara[dimIndex] = 1
    output = np.tile(A, repmatPara)

    return output


def freqOp(freq, marginWidth):
    freqSize = freq.shape
    D = freqSize[0]
    H = freqSize[1]
    W = freqSize[2]

    result = freq[marginWidth[0]:D - marginWidth[0], marginWidth[1]:H - marginWidth[1],
             marginWidth[2]:W - marginWidth[2]]

    return result


def inplace_fft(**varargin):
    # (buffer, marginWidth, ui, uj) for second derivative
    # (buffer, marginWidth, u) for first derivative
    if len(varargin) == 4:
        buffer = varargin[0]
        marginWidth = varargin[1]
        ui = varargin[2]
        uj = varargin[3]

        buffer = ui * uj * buffer  # Perform element-wised multiplication
        buffer = np.fft.ifft(buffer, axis='1')
        buffer = np.fft.ifft(buffer, axis='2')
        buffer = np.fft.ifft(buffer, axis='0')
        buffer = freqOp(buffer, marginWidth)

    elif len(varargin) == 3:
        buffer = varargin[0]
        marginWidth = varargin[1]
        u = varargin[2]

        buffer = u * buffer
        buffer = np.fft.ifft(buffer, axis='1')
        buffer = np.fft.ifft(buffer, axis='2')
        buffer = np.fft.ifft(buffer, axis='0')
        buffer = freqOp(buffer, marginWidth)

    return buffer


def oof_3d(image, radii, **options):
    marginwidth = [0, 0, 0]
    sizeImg = image.shape
    D = sizeImg[0]
    H = sizeImg[1]
    W = sizeImg[2]

    output = image[marginwidth[0]:D - marginwidth[0], marginwidth[1]:H - marginwidth[1],
             marginwidth[2]:W - marginwidth[2]]

    rtype = 0
    etype = 1
    ntype = 1
    pixelSpacing = [1, 1, 1]  # [z, x, y]
    sigma = min(pixelSpacing)
    alpha = 10
    tau = 0.01

    # set_trace()    # For debugging

    for key, value in options.items():
        if key == "spacing":
            pixelSpacing = value
            sigma = min(pixelSpacing)
        if key == "responsetype":
            rtype = value
        if key == "normalizationtype":
            ntype = value
        if key == "sigma":
            sigma = value
        if key == "useabsolute":
            etype = value
        if min(radii) < sigma & ntype > 0:
            print(
                'Sigma must be >= minimum range to enable the advanced normalization. The current setting falls back to options.normalizationtype=0, because of the undersize sigma.')
            ntype = 0

    imgfft = np.fft.fftn(image)

    ifft_cors = ifftshiftedcoormatrix(np.array(sizeImg))
    z = ifft_cors[0]
    x = ifft_cors[1]
    y = ifft_cors[2]

    z = z / D / pixelSpacing[0]
    x = x / H / pixelSpacing[1]
    y = y / W / pixelSpacing[2]
    radius = np.sqrt(z * z + x * x + y * y) + 1e-12

    for r in radii:
        print("Current OOF filter radii: " + str(r))

        whatIsIt = 4 / 3 * np.pi * pow(r, 3) / (sp_special.jv(1.5, 2 * np.pi * r * 1e-12) / pow(1e-12, 1.5)) / pow(r, 2)
        normalization = whatIsIt * pow(r / np.sqrt(2 * r * sigma - sigma * sigma), ntype)

        besseljBuffer = normalization * np.exp(-pow(sigma, 2) * 2 * np.pi * np.pi * radius * radius) / pow(radius, 1.5)
        besseljBuffer = (np.sin(2 * np.pi * r * radius) / (2 * np.pi * r * radius) - np.cos(
            2 * np.pi * r * radius)) * besseljBuffer * np.sqrt(1 / np.pi / np.pi / r / radius)
        besseljBuffer = besseljBuffer * imgfft

        q11 = freqOp(np.real(np.fft.ifftn(x * x * besseljBuffer)), marginwidth)
        q12 = freqOp(np.real(np.fft.ifftn(x * y * besseljBuffer)), marginwidth)
        q13 = freqOp(np.real(np.fft.ifftn(x * z * besseljBuffer)), marginwidth)
        q22 = freqOp(np.real(np.fft.ifftn(y * y * besseljBuffer)), marginwidth)
        q23 = freqOp(np.real(np.fft.ifftn(y * z * besseljBuffer)), marginwidth)
        q33 = freqOp(np.real(np.fft.ifftn(z * z * besseljBuffer)), marginwidth)

        eigenVal1, eigenVal2, eigenVal3 = eigenvaluefield33(q11, q12, q13, q22, q23, q33)

        maxEigenVal = eigenVal1.copy()
        condition = (eigenVal2 >= eigenVal1) & (eigenVal2 >= eigenVal3)
        maxEigenVal[condition] = eigenVal2[condition]
        condition = (eigenVal3 >= eigenVal1) & (eigenVal3 >= eigenVal2)
        maxEigenVal[condition] = eigenVal3[condition]

        minEigenVal = eigenVal1.copy()
        condition = (eigenVal2 <= eigenVal1) & (eigenVal2 <= eigenVal3)
        minEigenVal[condition] = eigenVal2[condition]
        condition = (eigenVal3 <= eigenVal1) & (eigenVal3 <= eigenVal2)
        minEigenVal[condition] = eigenVal3[condition]

        midEigenVal = eigenVal1 + eigenVal2 + eigenVal3 - maxEigenVal - minEigenVal

        lambda1 = maxEigenVal
        lambda2 = midEigenVal
        lambda3 = minEigenVal

        [V1x, V1y, V1z] = eigenvectorfield33(q11, q12, q13, q22, q23, q33, lambda3)

        lambda1ABS = np.abs(lambda1)
        lambda2ABS = np.abs(lambda2)
        lambda3ABS = np.abs(lambda3)

        Vesselness = lambda2 + lambda3

        """
        nonVesInd = (lambda2 > 0) | (lambda3 > 0)
        Vesselness[nonVesInd] = 2*np.maximum(lambda2[nonVesInd], lambda3[nonVesInd])
        Vesselness[~np.isfinite(Vesselness)] = 0
        Vesselness = 1 / (1 + np.exp(alpha*Vesselness))

        backgroundSup = lambda1ABS + lambda2ABS + lambda3ABS
        Vesselness[backgroundSup < tau] = 0
        """

        """
        ####
        # The following code ultize the absolute values of the eigenvalues, and thus the maximum one should be eigenvalues on the normal plane.
        ####
        maxEigenVal = eigenVal1
        minEigenVal = eigenVal1
        midEigenVal = eigenVal1 + eigenVal2 + eigenVal3

        maxEigenVal[np.abs(eigenVal2)>np.abs(maxEigenVal)] = eigenVal2[np.abs(eigenVal2)>np.abs(maxEigenVal)]
        minEigenVal[np.abs(eigenVal2)<np.abs(maxEigenVal)] = eigenVal2[np.abs(eigenVal2)<np.abs(maxEigenVal)]

        maxEigenVal[np.abs(eigenVal3)>np.abs(maxEigenVal)] = eigenVal2[np.abs(eigenVal3)>np.abs(maxEigenVal)]
        minEigenVal[np.abs(eigenVal3)<np.abs(maxEigenVal)] = eigenVal2[np.abs(eigenVal3)<np.abs(maxEigenVal)]

        midEigenVal = midEigenVal - maxEigenVal - minEigenVal

        lambda1 = maxEigenVal
        lambda2 = midEigenVal
        lambda3 = minEigenVal

        Vesselness = lambda1 + lambda2
        [V1x, V1y, V1z] = eigenvectorfield33(q11, q12, q13, q22, q23, q33, lambda1)
        """

        if r == radii[0]:
            oofv = Vesselness
            Lambda1 = lambda1
            Lambda2 = lambda2
            Lambda3 = lambda3

            whatScale = np.ones((sizeImg)) * r

            Voutx = V1x
            Vouty = V1y
            Voutz = V1z
        else:
            whatScale[Vesselness > oofv] = r

            condition = np.abs(Vesselness) > np.abs(oofv)
            Voutx[condition] = V1x[condition]
            Vouty[condition] = V1y[condition]
            Voutz[condition] = V1z[condition]
            oofv[condition] = Vesselness[condition]

            condition_eigen1 = np.abs(lambda1) > np.abs(Lambda1)
            Lambda1[condition_eigen1] = lambda1[condition_eigen1]
            condition_eigen2 = np.abs(lambda2) > np.abs(Lambda2)
            Lambda2[condition_eigen2] = lambda2[condition_eigen2]
            condition_eigen3 = np.abs(lambda3) > np.abs(Lambda3)
            Lambda1[condition_eigen3] = lambda3[condition_eigen3]

    return oofv, whatScale, Voutx, Vouty, Voutz
