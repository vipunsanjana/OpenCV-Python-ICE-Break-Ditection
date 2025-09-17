import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from utils.logger import logger  # import named logger

def detect_main_ice_break(video_path, output_dir, output_formats=None):
    """
    Process a video to detect the main ice break line in Lake Michigan with professional
    visualization, advanced measurements, and detailed analytics.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save output video
        output_formats: List of output video formats (extensions without dot)
                        Default is ['mp4', 'avi', 'mov', 'wmv']
    """

    logger.info("Starting ice break detection...")

    # Set default output formats if none provided
    if output_formats is None:
        output_formats = ['mp4', 'avi', 'mov', 'wmv']
        logger.info("No output formats specified. Using default formats: %s", output_formats)

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ensured: %s", output_dir)

    # Generate output base path without extension
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(output_dir, f"lake_michigan_ice_break_analysis_{timestamp}")
    logger.info("Base output file path: %s", output_base)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return
    logger.info("Successfully opened video: %s", video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video properties - Width: %d, Height: %d, FPS: %.2f, Total Frames: %d", 
                width, height, fps, total_frames)

    # Calculate the pixel-to-km ratio more accurately
    lake_width_km = 60  # estimated visible portion in km
    pixel_to_km_ratio = lake_width_km / width
    logger.info("Pixel-to-km ratio calculated: %.6f km/pixel", pixel_to_km_ratio)

    # Create a vibrant analysis color palette
    colors = {
        'primary': (255, 69, 0),       # Orange Red - main analysis line (eye-catching)
        'secondary': (50, 205, 50),    # Lime Green - secondary metrics
        'accent': (0, 255, 255),       # Cyan - highlights
        'highlight': (138, 43, 226),   # Blue Violet - special markers
        'background': (25, 25, 25),    # Almost black for sleek charts
        'text': (240, 240, 240),       # Off-white for readability
        'grid': (105, 105, 105),       # Dim gray grid lines
        'alert': (255, 20, 147),       # Deep Pink for alerts/warnings
        'water': (32, 99, 155)         # Deep blue (unchanged for sea)
    }
    logger.info("Custom color palette initialized for visualization")

    # Initialize video writers for different formats
    video_writers = {}
    codec_map = {
        'mp4': 'mp4v',
        'avi': 'XVID',
        'mov': 'mp4v',
        'wmv': 'WMV2'
    }

    for fmt in output_formats:
        if fmt in codec_map:
            fourcc = cv2.VideoWriter_fourcc(*codec_map[fmt])
            output_path = f"{output_base}.{fmt}"
            video_writers[fmt] = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info("Initialized %s writer: %s", fmt.upper(), output_path)

    # Create a named window for displaying processing
    cv2.namedWindow("Ice Break Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ice Break Detection", 640, 480)
    logger.info("Display window created for real-time visualization")

    # Initialize variables for tracking ice break measurements
    width_history = []       # Track detected ice width over time
    distance_history = []    # Track distance progression of ice break
    frame_count = 0          # Frame counter for analytics
    logger.info("Tracking variables initialized (width_history, distance_history, frame_count)")

    # Process each frame
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make a copy for drawing
        result_frame = frame.copy()
        
        # Add professional logo and title banner
        # Create title banner
        cv2.rectangle(result_frame, (0, 0), (width, 60), colors['background'], -1)
        cv2.putText(
            result_frame, "LAKE MICHIGAN ICE BREAK ANALYSIS", 
            (int(width/2) - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, colors['text'], 2, cv2.LINE_AA
        )
        
        # Add timestamp
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            result_frame, current_time, 
            (width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, colors['text'], 1, cv2.LINE_AA
        )
        
        # Convert to HSV color space to better isolate the blue water
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for blue color of the water in the crack
        lower_blue = np.array([90, 30, 40])
        upper_blue = np.array([130, 255, 255])
        
        # Create a mask for blue areas
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Apply advanced morphological operations to clean up the mask
        kernel = np.ones((7, 7), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a visualization of the mask
        mask_colored = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
        mask_colored[np.where((mask_colored == [255, 255, 255]).all(axis=2))] = colors['water']
        
        # Blend the mask with the original frame to show the detection area
        alpha = 0.3
        detection_blend = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)
        
        # Find the largest contour - this should be the main ice break
        if contours:
            # Sort contours by area, largest first
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Get the largest contour (the main ice break)
            main_contour = contours[0]
            
            # Draw the contour with a professional look
            cv2.drawContours(result_frame, [main_contour], -1, colors['accent'], 4)
            
            # Get a rectangle around the contour
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Draw bounding box with thickness of 4 and animated dashed effect
            dash_length = 20
            for i in range(0, 2*(w+h), 2*dash_length):
                # Top edge
                if i < w:
                    pt1 = (x + i, y)
                    pt2 = (x + min(i + dash_length, w), y)
                    cv2.line(result_frame, pt1, pt2, colors['primary'], 4, cv2.LINE_AA)
                # Right edge
                elif i < w + h:
                    pt1 = (x + w, y + (i - w))
                    pt2 = (x + w, y + min(i - w + dash_length, h))
                    cv2.line(result_frame, pt1, pt2, colors['primary'], 4, cv2.LINE_AA)
                # Bottom edge
                elif i < 2*w + h:
                    pt1 = (x + w - (i - w - h), y + h)
                    pt2 = (x + w - min(i - w - h + dash_length, w), y + h)
                    cv2.line(result_frame, pt1, pt2, colors['primary'], 4, cv2.LINE_AA)
                # Left edge
                else:
                    pt1 = (x, y + h - (i - 2*w - h))
                    pt2 = (x, y + h - min(i - 2*w - h + dash_length, h))
                    cv2.line(result_frame, pt1, pt2, colors['primary'], 4, cv2.LINE_AA)
            
            # Calculate accurate measurements
            leftmost = tuple(main_contour[main_contour[:, :, 0].argmin()][0])
            rightmost = tuple(main_contour[main_contour[:, :, 0].argmax()][0])
            topmost = tuple(main_contour[main_contour[:, :, 1].argmin()][0])
            bottommost = tuple(main_contour[main_contour[:, :, 1].argmax()][0])
            
            # Calculate actual width and height
            actual_width_pixels = rightmost[0] - leftmost[0]
            estimated_width_km = actual_width_pixels * pixel_to_km_ratio
            
            # Estimate distance from shore
            shore_distance_km = leftmost[0] * pixel_to_km_ratio
            
            # Store measurements for tracking
            width_history.append(estimated_width_km)
            distance_history.append(shore_distance_km)
            
            # Create a semi-transparent information panel
            info_panel_height = 180
            info_panel_width = 400
            info_panel = np.zeros((info_panel_height, info_panel_width, 3), dtype=np.uint8)
            info_panel[:, :] = colors['background']
            
            # Draw the semi-transparent panel on the frame
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (10, 70), (10 + info_panel_width, 70 + info_panel_height), colors['background'], -1)
            cv2.addWeighted(overlay, 0.8, result_frame, 0.2, 0, result_frame)
            
            # Add border to the panel
            cv2.rectangle(result_frame, (10, 70), (10 + info_panel_width, 70 + info_panel_height), colors['primary'], 2)
            
            # Add heading
            cv2.putText(
                result_frame, "ICE BREAK ANALYSIS", 
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors['secondary'], 2, cv2.LINE_AA
            )
            
            # Add horizontal separator line
            cv2.line(result_frame, (20, 105), (info_panel_width - 10, 105), colors['grid'], 1)
            
            # Add measurements with professional formatting
            cv2.putText(
                result_frame, f"Current Ice Width: {actual_width_pixels} px ({estimated_width_km:.2f} km)", 
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 1, cv2.LINE_AA
            )
            cv2.putText(
                result_frame, f"Distance from Shoreline: {shore_distance_km:.2f} km", 
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 1, cv2.LINE_AA
            )
            
            # Calculate and display area
            area_pixels = cv2.contourArea(main_contour)
            area_sq_km = area_pixels * (pixel_to_km_ratio ** 2)
            cv2.putText(
                result_frame, f"Ice Coverage Area: {area_sq_km:.2f} sq km", 
                (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 1, cv2.LINE_AA
            )
            
            # Add "PROCESSING ICE BREAK DATA" text with animated dots
            dots = "." * (frame_count % 4)
            cv2.putText(
                result_frame, f"PROCESSING ICE BREAK DATA{dots}", 
                (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['secondary'], 1, cv2.LINE_AA
            )
            
            # Draw lines to mark measurement points
            # Width line
            cv2.arrowedLine(result_frame, 
                          (leftmost[0], leftmost[1] - 30), 
                          (rightmost[0], rightmost[1] - 30), 
                          colors['highlight'], 2, cv2.LINE_AA, tipLength=0.02)
            cv2.arrowedLine(result_frame, 
                          (rightmost[0], rightmost[1] - 30), 
                          (leftmost[0], leftmost[1] - 30), 
                          colors['highlight'], 2, cv2.LINE_AA, tipLength=0.02)
            
            # Add marker dots at the measurement points
            cv2.circle(result_frame, leftmost, 5, colors['highlight'], -1)
            cv2.circle(result_frame, rightmost, 5, colors['highlight'], -1)
            cv2.circle(result_frame, topmost, 5, colors['secondary'], -1)
            cv2.circle(result_frame, bottommost, 5, colors['secondary'], -1)
            
            # Add trend mini-graph if we have enough data points
            if len(width_history) > 1:
                graph_width = 200
                graph_height = 100
                graph_x = width - graph_width - 20
                graph_y = 70
                
                # Create mini-graph background
                cv2.rectangle(result_frame, (graph_x, graph_y), 
                             (graph_x + graph_width, graph_y + graph_height), 
                             colors['background'], -1)
                cv2.rectangle(result_frame, (graph_x, graph_y), 
                             (graph_x + graph_width, graph_y + graph_height), 
                             colors['grid'], 1)
                
                # Display title
                cv2.putText(result_frame, "ICE WIDTH TREND", 
                           (graph_x + 10, graph_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1, cv2.LINE_AA)
                
                # Plot the width trend
                max_points = min(30, len(width_history))
                recent_widths = width_history[-max_points:]
                
                # Calculate plot scaling
                min_width = min(recent_widths) if recent_widths else 0
                max_width = max(recent_widths) if recent_widths else 1
                width_range = max_width - min_width if max_width > min_width else 1
                
                # Draw grid lines
                for i in range(5):
                    y_pos = graph_y + 30 + i * 15
                    cv2.line(result_frame, (graph_x + 5, y_pos), 
                            (graph_x + graph_width - 5, y_pos), 
                            colors['grid'], 1, cv2.LINE_AA)
                
                # Plot the trend line
                for i in range(1, len(recent_widths)):
                    x1 = graph_x + 10 + (i-1) * (graph_width - 20) // (max_points - 1)
                    y1 = graph_y + 90 - int(((recent_widths[i-1] - min_width) / width_range) * 60)
                    x2 = graph_x + 10 + i * (graph_width - 20) // (max_points - 1)
                    y2 = graph_y + 90 - int(((recent_widths[i] - min_width) / width_range) * 60)
                    cv2.line(result_frame, (x1, y1), (x2, y2), colors['secondary'], 2, cv2.LINE_AA)
                    cv2.circle(result_frame, (x2, y2), 2, colors['accent'], -1)
            
            # Add distance scale bar at bottom of frame
            scale_y = height - 30
            cv2.rectangle(result_frame, (0, scale_y - 20), (width, height), colors['background'], -1)
            
            # Draw km markers
            km_10_pixels = int(10 / pixel_to_km_ratio)
            for i in range(0, width, km_10_pixels):
                cv2.line(result_frame, (i, scale_y), (i, scale_y + 10), colors['text'], 1)
                if i > 0 and i % (2 * km_10_pixels) == 0:
                    cv2.putText(result_frame, f"{int(i * pixel_to_km_ratio)} km", 
                                (i - 20, scale_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, colors['text'], 1, cv2.LINE_AA)
            
            # Draw scale line
            cv2.line(result_frame, (0, scale_y), (width, scale_y), colors['text'], 1)
            
            # Add interesting feature: Detection confidence indicator
            confidence = min(100, int((area_pixels / (width * height)) * 1000))
            
            # Create a gauge display for confidence
            gauge_center_x = width - 100
            gauge_center_y = height - 100
            gauge_radius = 40
            
            # Draw gauge background
            cv2.ellipse(result_frame, (gauge_center_x, gauge_center_y), 
                       (gauge_radius, gauge_radius), 0, 135, 405, colors['background'], -1)
            cv2.ellipse(result_frame, (gauge_center_x, gauge_center_y), 
                       (gauge_radius, gauge_radius), 0, 135, 405, colors['grid'], 2)
            
            # Draw gauge level
            level_angle = 135 + (confidence * 2.7)  # Map 0-100 to 135-405 degrees
            cv2.ellipse(result_frame, (gauge_center_x, gauge_center_y), 
                       (gauge_radius, gauge_radius), 0, 135, level_angle, colors['secondary'], 4)
            
            # Add gauge text
            cv2.putText(result_frame, "CONFIDENCE", 
                       (gauge_center_x - 35, gauge_center_y + gauge_radius + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1, cv2.LINE_AA)
            cv2.putText(result_frame, f"{confidence}%", 
                       (gauge_center_x - 15, gauge_center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 1, cv2.LINE_AA)
        
        # Add frame counter
        cv2.putText(
            result_frame, f"Frame: {frame_count}/{total_frames}", 
            (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, colors['text'], 1, cv2.LINE_AA
        )
        
        # Display the processing frame
        cv2.imshow("Ice Break Detection", result_frame)
        
        # Update frame counter
        frame_count += 1
        
        # Check for key press (q to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Write the processed frame to all output videos
        for writer in video_writers.values():
            writer.write(result_frame)
    
    # After processing all frames, create a summary image with trend analysis
    if len(width_history) > 0:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(width_history, color='blue', linewidth=2)
            plt.title('Ice Break Width Trend')
            plt.xlabel('Frame Number')
            plt.ylabel('Width (km)')
            plt.grid(True)
            trend_image_path = f"{output_base}_trend.png"
            plt.savefig(trend_image_path)
            logger.info("Trend analysis image saved: %s", trend_image_path)
        except Exception as e:
            logger.error("Error generating trend analysis image: %s", e)
    
    # Release all resources properly
    cap.release()
    for writer in video_writers.values():
        writer.release()
    cv2.destroyAllWindows()
    logger.info("Released video capture, writers, and closed all OpenCV windows")

    # Log summary of created video files
    logger.info("Output Files Created:")
    for fmt in video_writers.keys():
        output_path = f"{output_base}.{fmt}"
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)  # Convert bytes to MB
        logger.info("- %s Video: %s (%.2f MB)", fmt.upper(), output_path, file_size_mb)

    # Log summary of generated trend image if it exists
    if len(width_history) > 0 and os.path.exists(f"{output_base}_trend.png"):
        trend_size_kb = os.path.getsize(f"{output_base}_trend.png") / 1024  # Convert bytes to KB
        logger.info("- Trend Image: %s (%.2f KB)", f"{output_base}_trend.png", trend_size_kb)

    # Final status log
    logger.info("Processing complete! All outputs successfully generated.")
