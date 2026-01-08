import numpy as np
from .butler import Butler as pButler


class OneCobra:
    calibModule = None

    # bit array for (x,y) to (theta, phi) convertion
    SOLUTION_OK           = 0x0001  # the solution is valid                     # noqa: E221
    IN_OVERLAPPING_REGION = 0x0002  # the position in overlapping region        # noqa: E221
    PHI_NEGATIVE          = 0x0004  # phi angle is negative(phi CCW limit < 0)  # noqa: E221
    PHI_BEYOND_PI         = 0x0008  # phi angle is beyond PI(phi CW limit > PI) # noqa: E221
    TOO_CLOSE_TO_CENTER   = 0x0010  # the position is too close to the center   # noqa: E221
    TOO_FAR_FROM_CENTER   = 0x0020  # the position is too far from the center   # noqa: E221

    def __init__(self, cobraIndex):
        """Return an object representing the single cobra corresponding to the 0-indexed index cobraIndex

        If cobraIndex is negative, return a nominal Cobra at the centre of the PFI
        """
        if OneCobra.calibModule is None:
            pbutler = pButler()
            OneCobra.calibModule = pbutler.get('moduleXml', moduleName='ALL', version='')

        self.cobraIndex = cobraIndex

        if cobraIndex < 0:
            self.center = 0 + 0*1j
            self.L1 = 2.4
            self.L2 = 2.4
            self.phiIn = -np.pi
            self.phiOut = 0
            self.tht0 = 0
            self.tht1 = 2*np.pi
        else:
            self.center = self.calibModule.centers[cobraIndex]
            self.L1 = self.calibModule.L1[cobraIndex]
            self.L2 = self.calibModule.L2[cobraIndex]
            self.phiIn = self.calibModule.phiIn[cobraIndex]
            self.phiOut = self.calibModule.phiOut[cobraIndex]
            self.tht0 = self.calibModule.tht0[cobraIndex]
            self.tht1 = self.calibModule.tht1[cobraIndex]

    def anglesToPositions(self, thetaAngles, phiAngles):
        """Convert the theta, phi angles to fiber positions.

        Parameters
        ----------
        thetaAngles: object
            A numpy array with the theta angles from CCW limit.
        phiAngles: object
            A numpy array with the phi angles from CCW limit.

        Returns
        -------
        numpy array
            A complex numpy array with the fiber positions.

        """
        if len(thetaAngles) != len(phiAngles):
            raise RuntimeError(f"number of theta angles ({len(thetaAngles)}) "
                               f"must match number of phi angles ({len(phiAngles)})")

        thtRange = (self.tht1 - self.tht0 + np.pi)%(2*np.pi) + np.pi
        invalid = (thetaAngles < 0) | (thetaAngles > thtRange)
        if np.sum(invalid) > 0:
            raise RuntimeError(f"Some theta angles are out of range: {thetaAngles[invalid]}")
        phiRange = self.phiOut - self.phiIn

        invalid = (phiAngles < 0) | (phiAngles > phiRange)
        if np.sum(invalid) > 0:
            raise RuntimeError(f"Some phi angles are out of range: {phiAngles[invalid]}")

        ang1 = self.tht0 + thetaAngles
        ang2 = self.phiIn + ang1 + phiAngles
        return self.center + self.L1*np.exp(1j*ang1) + self.L2*np.exp(1j*ang2)

    def positionsToAngles(self, positions):
        """Convert the fiber positions to theta, phi angles from CCW limit.

        Parameters
        ----------
        positions: numpy array
            A complex numpy array with the fiber positions.

        Returns
        -------
        tuple
            A python tuples with all the possible angles (theta, phi, flags).
            Since there are possible 2 phi solutions (since phi CCW<0 and CW>PI)
            so the dimensions of theta and phi are (len(cobraIndex), 2), the value
            np.nan indicates there is no solution. flags is a bit map.

            There are several different cases:
            - No solution: This means the distance from the given position to
              the center and two arm lengths(theta, phi) can't form a triangle.
              In this case, either TOO_CLOSE_TO_CENTER or TOO_FAR_FROM_CENTER
              is set. For TOO_CLOSE_TO_CENTER case, phi is set to 0 and for
              TOO_FAR_FROM_CENTER case, phi is set to PI, theta is set to
              the angle from the center to the given position for both cases.
            - Two phi solutions: This happens because the range of phi arms can
              be negative and beyond PI. When the measured phi angle is small or
              close to PI, this case may happen. The second solution is also
              calculated and returned. The bit PHI_NEGATIVE or PHI_BEYOND_PI is
              set in this situation. If this solution is within the hard stops,
              the bit SOLUTION_OK is set.
            - Theta overlapping region: Since theta arms can move beyond PI*2,
              so in the overlapping region(between two hard stops) we have two
              possible theta solutions. The bit IN_OVERLAPPING_REGION is set.
        """

        if np.isinf(self.L1 + self.L2):
            raise RuntimeError(f"{self.cobraIndex} has invalid L1 and or L2")

        # Calculate the cobra's rotation angles applying the law of cosines
        relativePositions = positions - self.center
        distance = np.abs(relativePositions)
        L1 = self.L1
        L2 = self.L2
        phiIn = self.phiIn + np.pi
        phiOut = self.phiOut + np.pi
        tht0 = self.tht0
        tht1 = self.tht1
        phi = np.full((len(positions), 2), np.nan)
        tht = np.full_like(phi, np.nan)
        flags = np.full_like(phi, 0, dtype='u2')

        for i in range(len(positions)):
            if distance[i] > L1 + L2:
                # too far away, return theta= spot angle and phi=PI
                flags[i][0] |= self.TOO_FAR_FROM_CENTER
                phi[i][0] = np.pi
                tht[i][0] = (np.angle(relativePositions[i]) - tht0)%(2*np.pi)
                if tht[i][0] <= (tht1 - tht0) % (2 * np.pi):
                    flags[i][0] |= self.IN_OVERLAPPING_REGION
                continue
            if distance[i] < np.abs(L1 - L2):
                # too close to center, theta is undetermined, return theta=spot angle and phi=0
                flags[i][0] |= self.TOO_CLOSE_TO_CENTER
                phi[i][0] = 0
                tht[i][0] = (np.angle(relativePositions[i]) - tht0) % (2 * np.pi)
                if tht[i][0] <= (tht1 - tht0) % (2 * np.pi):
                    flags[i][0] |= self.IN_OVERLAPPING_REGION
                continue

            ang1 = np.arccos((L1**2 + L2**2 - distance[i]**2)/(2*L1*L2))
            ang2 = np.arccos((L1**2 + distance[i]**2 - L2**2)/(2*L1*distance[i]))

            # the regular solutions, phi angle is between 0 and pi, no checking for phi hard stops
            flags[i][0] |= self.SOLUTION_OK
            phi[i][0] = ang1 - phiIn
            tht[i][0] = (np.angle(relativePositions[i]) + ang2 - tht0) % (2 * np.pi)
            # check if tht is within two theta hard stops
            if tht[i][0] <= (tht1 - tht0) % (2 * np.pi):
                flags[i][0] |= self.IN_OVERLAPPING_REGION

            # check if there are additional solutions
            if ang1 <= np.pi/2 and ang1 > 0:
                if phiIn <= -ang1:
                    flags[i][1] |= self.SOLUTION_OK
                flags[i][1] |= self.PHI_NEGATIVE
                # phiIn < 0
                phi[i][1] = -ang1 - phiIn
                tht[i][1] = (np.angle(relativePositions[i]) - ang2 - tht0) % (2 * np.pi)
                # check if tht is within two theta hard stops
                if tht[i][1] <= (tht1 - tht0) % (2 * np.pi):
                    flags[i][1] |= self.IN_OVERLAPPING_REGION
            elif ang1 > np.pi/2 and ang1 < np.pi:
                if phiOut >= 2 * np.pi - ang1:
                    flags[i][1] |= self.SOLUTION_OK
                flags[i][1] |= self.PHI_BEYOND_PI
                # phiOut > np.pi
                phi[i][1] = 2 * np.pi - ang1 - phiIn
                tht[i][1] = (np.angle(relativePositions[i]) - ang2 - tht0) % (2 * np.pi)
                # check if tht is within two theta hard stops
                if tht[i][1] <= (tht1 - tht0) % (2 * np.pi):
                    flags[i][1] |= self.IN_OVERLAPPING_REGION

        return (tht, phi, flags)
