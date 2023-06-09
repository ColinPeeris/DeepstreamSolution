import cv2


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


class ROI:
    def __init__(self, roi_points):
        self.roi_points = roi_points

    def update_roi(self, roi_points):
        self.roi_points = roi_points

    def test(self):
        # Driver code
        self.roi_points = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
        p = Point(5, 3)

        # Function call
        if (self.checkInside(p)):
            print("Point is inside.")
        else:
            print("Point is outside.")

    def onLine(self, l1, p):
        # Check whether p is on the line or not
        if (
            p.x <= max(l1.p1.x, l1.p2.x)
            and p.x <= min(l1.p1.x, l1.p2.x)
            and (p.y <= max(l1.p1.y, l1.p2.y) and p.y <= min(l1.p1.y, l1.p2.y))
        ):
            return True
        return False

    def direction(self, a, b, c):
        val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)
        if val == 0:
            # Colinear
            return 0
        elif val < 0:
            # Anti-clockwise direction
            return 2
        # Clockwise direction
        return 1

    def isIntersect(self, l1, l2):
        # Four direction for two lines and points of other line
        dir1 = self.direction(l1.p1, l1.p2, l2.p1)
        dir2 = self.direction(l1.p1, l1.p2, l2.p2)
        dir3 = self.direction(l2.p1, l2.p2, l1.p1)
        dir4 = self.direction(l2.p1, l2.p2, l1.p2)

        # When intersecting
        if dir1 != dir2 and dir3 != dir4:
            return True

        # When p2 of line2 are on the line1
        if dir1 == 0 and self.onLine(l1, l2.p1):
            return True

        # When p1 of line2 are on the line1
        if dir2 == 0 and self.onLine(l1, l2.p2):
            return True

        # When p2 of line1 are on the line2
        if dir3 == 0 and self.onLine(l2, l1.p1):
            return True

        # When p1 of line1 are on the line2
        if dir4 == 0 and self.onLine(l2, l1.p2):
            return True

        return False

    def checkInside(self, p):
        poly = self.roi_points
        n = len(poly)
        # When polygon has less than 3 edge, it is not polygon
        if n < 3:
            return False

        # Create a point at infinity, y is same as point p
        exline = line(p, Point(9999, p.y))
        count = 0
        i = 0
        while True:
            # Forming a line from two consecutive points of poly
            side = line(poly[i], poly[(i + 1) % n])
            if self.isIntersect(side, exline):
                # If side is intersects ex
                if (self.direction(side.p1, p, side.p2) == 0):
                    return self.onLine(side, p)
                count += 1

            i = (i + 1) % n
            if i == 0:
                break

        # When count is odd
        return count & 1

    def drawROI(self, image):
        # Green color in BGR
        color = (0, 255, 0)
        # Line thickness of 9 px
        thickness = 9

        for i in range(len(self.roi_points) - 1):
            start_point = (int(self.roi_points[i].x), int(self.roi_points[i].y))
            end_point = (int(self.roi_points[i + 1].x), int(self.roi_points[i + 1].y))
            image = cv2.line(image, start_point, end_point, color, thickness)

        # close the loop
        last_point = (int(self.roi_points[-1].x), int(self.roi_points[-1].y))
        first_point = (int(self.roi_points[0].x), int(self.roi_points[0].y))
        image = cv2.line(image, last_point, first_point, color, thickness)

        return image
