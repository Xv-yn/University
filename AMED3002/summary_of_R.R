
# Get Current Working Directory
getwd()

# # Set Current Working Directory 
# setwd("~/Desktop/University/AMED3002/") 

# # Read From Chosen File
# data1 <- read.csv(file.choose(), header=T)
                                    # header checks if the first line in the file contains variable names
# Read From Specified File
data1 <- read.csv("~/Desktop/University/AMED3002/student_sleep_patterns.csv", header=T)

data1

library(ggplot2)
library(dplyr)

p <- ggplot(data = data1, aes(x = Sleep_Duration, y = Study_Hours, col = Age))
                          # Aesthetics Layer         Color
p + labs(title = "Sleeping Study Plot", subtitle = "subtitle goes here", caption = "caption here") + # Labels (Title, Subtitle, Caption, Tag)

# Geometric Layer

    geom_point() + 
  # geom_bar() +
  # geom_violin() +
  # geom_histogram() +
  # geom_line() +
  # geom_boxplot() +

  # Note that you can stack the geom_ functions
  # See: https://r-graph-gallery.com/ggplot2-package.html for more details

# Facet Layer
  
  facet_grid(rows = vars(Gender)) +
  # facet_grid(cols = vars(PhoneReach))

  # Similarly, you can stack facet_
  # See: http://www.cookbook-r.com/Graphs/Facets_(ggplot2)/


# Statistical Layer
  stat_midwest_select <- midwest[midwest$poptotal > 350000 & 
                            midwest$poptotal <= 500000 & 
                            midwest$area > 0.01 & 
                            midwest$area < 0.1, ]smooth(method ="lm", col = "red") + 

  # Similarly, you can stack stat_ 
  # See: https://ggplot2.tidyverse.org/reference/layer_stats.html

# Coordinate Layer
  coord_cartesian(xlim = c(3.5, 9.5))

  # Similarly, you can stack coord_ 
  # See: https://ggplot2.tidyverse.org/reference/layer_stats.html 


# See: https://r-statistics.co/Top50-Ggplot2-Visualizations-MasterList-R-Code.html#Bubble%20Plot
