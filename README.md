
<!-- GETTING STARTED -->
## Getting Started

This project is used to capture daily images using an OAK-d camera, which is maunted in a farm bot.

### Prerequisites

Torun the project you will need some packages installed for example `opencv-python`.
  ```sh
  pip install opencv-python
  ```

### Installation

Below is the explanation, what you should do to run the application

1. Intall python in your machine if you haven't already

2. Clone the repo
   ```sh
   git clone [https://github.com/your_username_/Project-Name.git](https://github.com/aarapi/robot-pharm-daily-images.git)
   ```
3. Install python packages if are needed
   ```sh
   pip install
   ```
4. Enter list of times that you want to take pictures during the day in `config/config.xml` file
   ```sh
   <Daily_Time_Image>13:06:20</Daily_Time_Image>
   ```
   you can add as many as you want
5. Run python script `gardenBot.py`
   ```sh
   python gardenBot.py
   ```
    
