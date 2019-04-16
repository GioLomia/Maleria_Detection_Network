train_dir <- "C:\\Users\\lomiag\\Desktop\\Malaria Detection ZIP\\Malaria Detection\\Train"
val_dir<- "C:\\Users\\lomiag\\Desktop\\Malaria Detection ZIP\\Malaria Detection\\Valid"
test_dir <-"C:\\Users\\lomiag\\Desktop\\Malaria Detection ZIP\\Malaria Detection\\Test"

## Let's build a convnet model!

library(keras)

## data augmentation configuration

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40, # randomly rotate pictures within a range
  width_shift_range = 0.2, # randomly translate picture as a % of width and height
  height_shift_range = 0.2,
  shear_range = 0.2, # randoml apply shearing transformations
  zoom_range = 0.2, # randomly apply zoom as a % of picture size
  horizontal_flip = TRUE, # randomly flipping half the images horizontally; when there is no assumption of horizontal asymmetry
  fill_mode = "nearest" # mode for filling in pixels as a result of previous transformations
)


## displaying randomly augmented training images

## new convnet that includes dropouts

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 6, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(100, 100, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 12, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 24, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 412, activation = "relu") %>%
  layer_dropout(rate=0.3)%>%
  layer_dense(units = 100, activation =  "relu")%>%
  layer_batch_normalization()%>%
  layer_dense(units = 1, activation = "sigmoid")

model

model %>% compile(
  optimizer = optimizer_adam(),
  loss = "binary_crossentropy",
  metrics = c("acc")
)

# random image augmentation

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(          # validation data is not augmented, only training data
  train_dir,                           # target directory              
  datagen,                             # data generator                
  target_size = c(100, 100),           # resize images                 
  batch_size = 32,
  class_mode = "binary"                # binary classification (cats OR dogs)             
)

validation_generator <- flow_images_from_directory(
  val_dir,
  test_datagen,
  target_size = c(100, 100),
  batch_size = 32,
  class_mode = "binary"   
)
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(100, 100),
  batch_size = 32,
  class_mode = "binary"   
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 50,
  epochs = 2,
  validation_data = validation_generator,
  validation_steps = 50
)

score<-model%>%evaluate_generator(generator=test_generator,steps = 50)
score
