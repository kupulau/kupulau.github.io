---
title: Installing ds9 on Mac
category: general
tags:
  - mac
  - astronomy
url: https://velog.io/@kupulau/Installing-ds9-on-Mac
created_at: 2024-10-23
related_notes:
---

(ref: 2024 Oct 23rd)
Here I download ds9 via Darwin X11 on Sonoma 14 Apple Silicon.

#### 1. Go to SAOds9 official download [site](https://sites.google.com/cfa.harvard.edu/saoimageds9/download?authuser=0)

#### 2. Download one of the most suitable file for your environment
1) Select the port (Aqua or Darwin X11)
2) Check your MacOS (Sonoma or Intel or etc...)

#### 3. Unpack the file
`% tar -xvzf tar -xvzf darwin<os><arch>.<version>.tar.gz`

#### 4. Move the file to your local bin directory
`% mv ds9 ds9.zip $HOME/bin/.`

If you don't know where your local bin directory is, type `echo $PATH`.

**Now you can use ds9 on your Mac!**

<br>

#### Security issue
You may encounter the message "unable to open" due to the security issue. In that case, change the setting at the privacy & security tap on your Mac.