"""Contact Sheet Generator for Multi-View Image Datasets.

Assembles multi-view image sequences into a structured grid layout with:
- Preserved aspect ratios (letterboxed/centered thumbnails)
- Filename labels below each thumbnail
- Clean contrast-rich aesthetic suitable for research inspection
"""

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


def create_contact_sheet(
    image_paths: List[Path | str],
    labels: Optional[List[str]] = None,
    columns: int = 8,
    thumb_box_size: Tuple[int, int] = (220, 293),  # (width, height) preserving 3:4 portrait
    cell_padding: int = 12,
    label_height: int = 36,
    bg_color: Tuple[int, int, int] = (30, 32, 38),  # Dark slate BGR
    cell_bg_color: Tuple[int, int, int] = (42, 45, 54),  # Slightly lighter slate
    border_color: Tuple[int, int, int] = (60, 64, 76),
    text_color: Tuple[int, int, int] = (220, 225, 235),
    header_title: Optional[str] = "THU-MVS Dog_RGB Dataset - Contact Sheet (72 Views)",
) -> np.ndarray:
    """Generates a high-resolution contact sheet image grid.

    Args:
        image_paths: List of paths to images to display.
        labels: Optional custom labels for each thumbnail. Defaults to filename.
        columns: Number of grid columns (default 8).
        thumb_box_size: (max_w, max_h) boundary box for each thumbnail.
        cell_padding: Outer padding inside each cell.
        label_height: Vertical pixel height allocated for the text label.
        bg_color: Canvas background color (BGR).
        cell_bg_color: Individual thumbnail card background color (BGR).
        border_color: Card boundary outline color (BGR).
        text_color: Label text font color (BGR).
        header_title: Title text string for the top banner, or None.

    Returns:
        np.ndarray: BGR image array containing the rendered contact sheet.
    """
    total_images = len(image_paths)
    if total_images == 0:
        raise ValueError("Cannot create contact sheet: image_paths list is empty.")

    if labels is None:
        labels = [Path(p).name for p in image_paths]

    thumb_w, thumb_h = thumb_box_size
    cell_w = thumb_w + (2 * cell_padding)
    cell_h = thumb_h + (2 * cell_padding) + label_height

    rows = int(np.ceil(total_images / columns))

    header_height = 70 if header_title else 0
    canvas_w = columns * cell_w
    canvas_h = header_height + (rows * cell_h)

    # Initialize canvas
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    # Render header banner if specified
    if header_title:
        # Title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size, _ = cv2.getTextSize(header_title, font, font_scale, thickness)
        text_x = (canvas_w - text_size[0]) // 2
        text_y = 44
        cv2.putText(
            canvas,
            header_title,
            (text_x, text_y),
            font,
            font_scale,
            (240, 240, 245),
            thickness,
            cv2.LINE_AA,
        )
        # Separator line
        cv2.line(
            canvas,
            (20, header_height - 10),
            (canvas_w - 20, header_height - 10),
            border_color,
            1,
            cv2.LINE_AA,
        )

    # Render each cell
    for idx, (path_item, label_text) in enumerate(zip(image_paths, labels)):
        col_idx = idx % columns
        row_idx = idx // columns

        c_x = col_idx * cell_w
        c_y = header_height + (row_idx * cell_h)

        # Draw cell card background and border
        card_x1 = c_x + 4
        card_y1 = c_y + 4
        card_x2 = c_x + cell_w - 4
        card_y2 = c_y + cell_h - 4
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), cell_bg_color, -1)
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), border_color, 1)

        # Load image safely
        p = Path(path_item)
        img = None
        if p.exists():
            with open(p, "rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        thumb_box_x = c_x + cell_padding
        thumb_box_y = c_y + cell_padding

        if img is not None:
            # Preserve original aspect ratio: fit within (thumb_w, thumb_h)
            h, w = img.shape[:2]
            scale = min(thumb_w / w, thumb_h / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Center thumbnail inside the thumbnail box
            x_offset = thumb_box_x + (thumb_w - new_w) // 2
            y_offset = thumb_box_y + (thumb_h - new_h) // 2

            canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
        else:
            # Fallback placeholder for unreadable image
            cv2.putText(
                canvas,
                "UNREADABLE",
                (thumb_box_x + 20, thumb_box_y + thumb_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # Render label text centered below thumbnail box
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        label_scale = 0.48
        label_thick = 1
        lbl_size, _ = cv2.getTextSize(label_text, label_font, label_scale, label_thick)
        lbl_x = c_x + (cell_w - lbl_size[0]) // 2
        lbl_y = c_y + cell_h - cell_padding - 8

        cv2.putText(
            canvas,
            label_text,
            (lbl_x, lbl_y),
            label_font,
            label_scale,
            text_color,
            label_thick,
            cv2.LINE_AA,
        )

    return canvas
