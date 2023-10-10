


# Load necessary packages
library(shiny)
library(shinythemes)
library(tidyverse)

# Define the UI
ui <- fluidPage(
  titlePanel("🔴🟡🟢Diabetes Predictor: Know Your Risk ❗️"),
  theme = shinytheme("flatly"),
  sidebarLayout(
    sidebarPanel(
      # Define text input fields with labels and default values
      numericInput("Glucose", "Glucose value🍦 :", min = 10, max = 120, value = 120),
      numericInput("BMI", "BMI⏲ :", min = 10, max = 50, value = 30),
      numericInput("Age", "Age in years⌛️ :", min = 20, max = 80, value = 40),
      numericInput("Diabetespedigree", "Pedigree score 💉 :", min = 0.1, max = 2, value = 1.5),
      numericInput("Bloodpressure", "Blood pressure🌡:", min = 60, max = 140, value = 80),
      numericInput("Pregnancies", "Pregnancies🤰:", min = 0, max = 20, value = 4),
      numericInput("Insulin", "Enter the insulin level💊:", min = 0, max = 200, value = 60),
      numericInput("Skinthickness", "Skin thickness🔬:", min = 0, max = 50, value = 20),
      actionButton("predictButton", "Predict🔎")
    ),
    mainPanel(
      h3("The patient is....⚉", align = "center"),
      br(),
      textOutput("pred"),
      br(),
      img(src = "diabetes.png", height = 250, width = 250)
    )
  )
)

# Define the server logic
server <- function(input, output) {
  # Load the pre-trained model
  model <- readRDS("randomforest_model.rds")
  
  
  
  observeEvent(input$predictButton, {
    # Process input data (center and scale)
    input_data <- data.frame(
      Glucose = input$Glucose,
      BMI = input$BMI,
      Age = input$Age,
      DiabetesPedigreeFunction = input$Diabetespedigree,
      BloodPressure = input$Bloodpressure,
      Pregnancies = input$Pregnancies,
      Insulin = input$Insulin,
      SkinThickness = input$Skinthickness)

    
    # Make predictions
    pred <- predict(model, input_data)
    output$pred <- renderText({
      if (pred == 1) {
        return("Diabetic")
      } else {
        return("Not diabetic")
      }
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)


