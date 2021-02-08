from bbox_utils.point_cloud import PointCloud


def main():
    pcd = PointCloud.load_from_file("./assets/demo_pointcloud.pcd")
    pcd.display()


if __name__ == "__main__":
    main()
