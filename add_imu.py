#!/usr/bin/env python

import piexif
import piexif.helper
import argparse

def main(filename):
    image_path = filename

    # Roll, pitch, yaw (degrees) — same order/names as SU-WaterCam add_metadata.py
    roll = 1.5
    pitch = -8.0   # slight downward tilt
    yaw = 120.0   # facing southeast-ish

    exif_dict = piexif.load(image_path)

    # Match SU-WaterCam format: "Roll {roll} Pitch {pitch} Yaw {yaw}"
    user_comment = piexif.helper.UserComment.dump(f"Roll {roll} Pitch {pitch} Yaw {yaw}")
    if "Exif" not in exif_dict:
        exif_dict["Exif"] = {}
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment

    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)
    print("Added IMU (Roll/Pitch/Yaw) to UserComment")

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Add IMU values to an image")
     parser.add_argument('filename')
     args = parser.parse_args()
     main(args.filename)
