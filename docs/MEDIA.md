# Media

The repository includes two photographs, two video preview images and two silent demonstrations. Original media remain outside the repository in the project archive.

| Repository file | Description |
| --- | --- |
| [robot-prototype.jpg](../media/images/robot-prototype.jpg) | Robot and mock palm trunks in the laboratory test area |
| [tree-detection.jpg](../media/images/tree-detection.jpg) | Camera screenshot showing detected mock trunks |
| [navigation-preview.jpg](../media/images/navigation-preview.jpg) | Still frame from the navigation demonstration |
| [dashboard-preview.jpg](../media/images/dashboard-preview.jpg) | Still frame from the operator-interface demonstration |
| [robot-navigation.mp4](../media/videos/robot-navigation.mp4) | Approximately 75 seconds of robot movement through the laboratory course |
| [dashboard-perception.mp4](../media/videos/dashboard-perception.mp4) | Approximately 63 seconds showing the dashboard, perception view and map |

## Preparation

- The original HEIC photo was exported to a broadly compatible JPEG, oriented correctly and resized to fit within 1600 × 1600 pixels.
- Both exported photographs and both extracted preview images contain no EXIF metadata. GPS coordinates and device/date metadata from the HEIC original were not carried into the publication copy.
- Audio was removed from both videos. Source video container metadata and chapters were removed during remuxing.
- H.264 video packets were copied without re-encoding.
- Preview images are extracted frames from the actual supplied recordings.
- The recordings show historical laboratory demonstrations, not validation results for every source file in this repository.

GPS metadata removal does not alter text already visible inside a photograph or video. The dashboard demonstration includes its original screen content and local robot connection settings.

## File sizes and GitHub

The publication videos are approximately 10.4 MiB and 8.7 MiB. Both fit in a normal Git repository and are below GitHub's 25 MiB per-file browser-upload limit and 100 MiB Git limit. Git LFS is unnecessary for this media set. See [GitHub file-upload limits](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

The README uses JPG previews linked to MP4 files. It does not depend on inline HTML video playback. GitHub comment/issue video attachments have separate limits from tracked repository files, so a repository MP4 link is used here. [GitHub attachment documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)

[manifest.json](../media/manifest.json) records exact file sizes, approximate source durations, codec and audio status.
