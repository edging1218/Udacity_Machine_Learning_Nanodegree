library('ggmap')
library('ggplot2')

data = read.csv('input/Crimes_-_2016.csv', header=T)
newdata = na.omit(data)
map = get_map('Chicago', zoom = 11)
ggmap(map)
plotdata = data.frame(newdata$Primary.Type, newdata$Latitude, newdata$Longitude)
colnames(plotdata) = c('type', 'Latitude', 'Longitude')
contours = stat_density2d(
  aes(x = Longitude, y = Latitude, fill = ..level.., alpha = ..level..),
  size = 0.1,
  data = plotdata,
  n = 200,
  geom = 'polygon')
ggmap(map)+ contours
s
