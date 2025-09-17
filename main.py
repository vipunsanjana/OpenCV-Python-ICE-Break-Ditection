from lake_ice_break.detection import detect_main_ice_break
from utils.logger import logger  # import named logger

def main():
    video_path = r"data\ice.mp4"
    output_dir = r"output"
    output_formats = ['mp4', 'avi', 'mov', 'wmv']
    
    logger.info("Starting ice break detection...")
    detect_main_ice_break(video_path, output_dir, output_formats)
    logger.info("Ice break detection completed.")

if __name__ == "__main__":
    main()
