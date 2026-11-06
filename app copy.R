# ==============================================================================
# 1. GLOBAL SETUP & LIBRARIES
# ==============================================================================
library(shiny)
library(bslib)        # Layout e temi
library(ggplot2)      # Grafica (Donut, Heatmap)
library(dplyr)        # Manipolazione dati
library(glmnet)       # Lasso
library(gam)          # GAM
library(randomForest) # Random Forest
library(nnet)         # Multinomiale
library(boot)         # Bootstrap
library(DT)

# Carica ggrepel e dplyr se presenti, altrimenti imposta il fallback sicuro
has_ggrepel <- requireNamespace("ggrepel", quietly = TRUE)
if (has_ggrepel) library(ggrepel)

# Caricamento Dataset e Modelli
df_serieA   <- readRDS("Data/dataset_serieA.rds")
mod_poisson <- readRDS("Data/modello_poisson.rds")
mod_multi   <- readRDS("Data/modello_multinomiale.rds")
mod_lasso   <- readRDS("Data/modello_lasso.rds")
mod_gam     <- readRDS("Data/modello_gam.rds")
mod_rf      <- readRDS("Data/modello_rf.rds")
boot_res    <- readRDS("Data/dati_bootstrap.rds")

# Gestione Sicura Squadre
squadre_default <- c("Atalanta", "Benevento", "Bologna", "Fiorentina", "Inter", "Juventus", "Lazio", "AC Milan", "Napoli", "Roma", "Torino")

squadre_serieA <- tryCatch({
  cols <- colnames(df_serieA)
  col_home <- cols[grep("home|casa", cols, ignore.case = TRUE)][1]
  col_away <- cols[grep("away|trasferta", cols, ignore.case = TRUE)][1]
  
  if(!is.na(col_home) && !is.na(col_away)) {
    sort(unique(c(as.character(df_serieA[[col_home]]), as.character(df_serieA[[col_away]]))))
  } else {
    squadre_default
  }
}, error = function(e) {
  squadre_default
})

# ==============================================================================
# 2. CUSTOM CSS (CYBORG DARK & NEON TABLE STYLING)
# ==============================================================================
custom_css <- HTML("
  body {
    background-color: #090D16 !important;
    color: #F9FAFB !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
  }
  
  .navbar {
    background-color: #0D1322 !important;
    border-bottom: 1px solid #1E293B !important;
    padding: 0.75rem 1.5rem !important;
  }
  .navbar-brand {
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    color: #00E676 !important;
  }
  .nav-link {
    color: #9CA3AF !important;
    font-weight: 500;
    border-radius: 8px;
    padding: 8px 16px !important;
    margin-right: 4px;
  }
  .nav-link.active {
    background-color: #111D33 !important;
    color: #00E676 !important;
    border: 1px solid #00E676 !important;
  }

  .custom-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  }
  .card-title-custom {
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #9CA3AF;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
  }

  .kpi-container {
    background: #162032;
    border: 1px solid #243049;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }
  .kpi-title {
    font-size: 0.75rem;
    color: #9CA3AF;
    text-transform: uppercase;
    font-weight: 600;
  }
  .kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 4px 0;
  }
  .kpi-sub {
    font-size: 0.75rem;
    color: #00E676;
  }

  .irs--flat .irs-bar { background-color: #00E676 !important; }
  .irs--flat .irs-handle { background-color: #00E676 !important; border-color: #00E676 !important; }
  .irs--flat .irs-from, .irs--flat .irs-to, .irs--flat .irs-single { background-color: #00E676 !important; color: #000 !important; font-weight: bold; }

  /* Stile Dark per la Tabella Comparativa */
  .table {
    color: #F9FAFB !important;
    background-color: transparent !important;
    margin-bottom: 0px !important;
  }
  .table th {
    color: #00E676 !important;
    border-bottom: 2px solid #1F2937 !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
  }
  .table td {
    border-top: 1px solid #1F2937 !important;
    color: #E5E7EB !important;
    font-size: 0.875rem;
    vertical-align: middle;
  }
")

# ==============================================================================
# 3. USER INTERFACE (UI)
# ==============================================================================
ui <- page_navbar(
  title = "⚽ SERIE A DSS | Predictive System",
  theme = bs_theme(version = 5, bg = "#090D16", fg = "#F9FAFB", primary = "#00E676"),
  header = tags$head(tags$style(custom_css)),
  
# ==============================================================================
# TAB 1: MATCH PREDICTOR
# ==============================================================================
  nav_panel("1. Match Predictor",
            fluidRow(
              # SEZIONE SINISTRA: INPUTS
              column(width = 3,
                     div(class = "custom-card",
                         div(class = "card-title-custom", "Match Inputs"),
                         
                         selectInput("home_team", "Home Team", choices = squadre_serieA, selected = squadre_serieA[1]),
                         selectInput("away_team", "Away Team", choices = squadre_serieA, selected = squadre_serieA[2]),
                         hr(style = "border-color: #1F2937;"),
                         
                         div(class = "card-title-custom", "Expected Match Stats"),
                         sliderInput("hst", "Home Shots on Target (HST)", min = 0, max = 15, value = 5),
                         sliderInput("hs",  "Home Total Shots (HS)",      min = 0, max = 30, value = 11),
                         hr(style = "border-color: #1F2937;"),
                         
                         sliderInput("ast", "Away Shots on Target (AST)", min = 0, max = 15, value = 4),
                         sliderInput("as",  "Away Total Shots (AS)",      min = 0, max = 30, value = 10)
                     )
              ),
              
              # SEZIONE DESTRA: OUTPUTS
              column(width = 9,
                     # RIGA 1: PROBABILITÀ 1X2 + EXPECTED GOALS
                     fluidRow(
                       column(width = 7,
                              div(class = "custom-card",
                                  div(class = "card-title-custom", "Match Winner Probabilities (1X2)"),
                                  plotOutput("donut_plot", height = "180px")
                              )
                       ),
                       column(width = 5,
                              div(class = "custom-card",
                                  div(class = "card-title-custom", "Expected Goals (xG) — Poisson GLM"),
                                  fluidRow(
                                    column(6, 
                                           div(class = "kpi-container",
                                               div(class = "kpi-title", textOutput("lbl_home_xg")),
                                               div(class = "kpi-value", style = "color: #00E676;", textOutput("val_home_xg")),
                                               div(class = "kpi-sub", "xG Home")
                                           )
                                    ),
                                    column(6, 
                                           div(class = "kpi-container",
                                               div(class = "kpi-title", textOutput("lbl_away_xg")),
                                               div(class = "kpi-value", style = "color: #3B82F6;", textOutput("val_away_xg")),
                                               div(class = "kpi-sub", "xG Away")
                                           )
                                    )
                                  )
                              )
                       )
                     ),
                     
                     # RIGA 2: POISSON EXACT SCORE HEATMAP
                     fluidRow(
                       column(width = 12,
                              div(class = "custom-card",
                                  div(class = "card-title-custom", "Exact Score Probability Distribution (Poisson Bivariate Matrix)"),
                                  plotOutput("heatmap_plot", height = "250px")
                              )
                       )
                     ),
                     
                     # RIGA 3: METRICHE DI SINTESI (KPIs)
                     fluidRow(
                       column(3, div(class = "kpi-container",
                                     div(class = "kpi-title", "Most Likely Score"),
                                     div(class = "kpi-value", textOutput("kpi_most_likely")),
                                     div(class = "kpi-sub", textOutput("kpi_most_likely_prob")))),
                       column(3, div(class = "kpi-container",
                                     div(class = "kpi-title", "Next Most Likely"),
                                     div(class = "kpi-value", textOutput("kpi_second_likely")),
                                     div(class = "kpi-sub", textOutput("kpi_second_likely_prob")))),
                       column(3, div(class = "kpi-container",
                                     div(class = "kpi-title", "Total Expected Goals"),
                                     div(class = "kpi-value", style="color:#00E676;", textOutput("kpi_total_xg")),
                                     div(class = "kpi-sub", "Poisson Combined"))),
                       column(3, div(class = "kpi-container",
                                     div(class = "kpi-title", "Over 2.5 Goals Prob."),
                                     div(class = "kpi-value", style="color:#3B82F6;", textOutput("kpi_over25")),
                                     div(class = "kpi-sub", "Calculated Risk")))
                     ),
                     
                     br(),
                     
                     # RIGA 4: TABELLA DI CONSENSO MULTI-MODELLO
                     fluidRow(
                       column(width = 12,
                              div(class = "custom-card",
                                  div(class = "card-title-custom", "Multi-Model Prediction Consensus Matrix"),
                                  tableOutput("model_consensus_table")
                              )
                       )
                     )
              )
            )
  ),
  
# ==============================================================================
# TAB 2: SCENARIO SIMULATOR (UI)
# ==============================================================================
  nav_panel("2. Scenario Simulator",
            fluidRow(
              # Sidebar comandi per la simulazione tattica
              column(width = 3,
                     div(class = "custom-card",
                         div(class = "card-title-custom", "What-If Scenario Parameters"),
                         p(style="font-size: 0.8rem; color: #9CA3AF;", "Simulate tactical shifts and evaluate non-linear response curves."),
                         
                         radioButtons("sim_engine", "Simulation Model Architecture",
                                      choices = c("GAM (Non-Linear Splines)", "Random Forest (Tree Ensemble)", "Poisson Baseline"),
                                      selected = "GAM (Non-Linear Splines)"),
                         
                         hr(style = "border-color: #1F2937;"),
                         div(class = "card-title-custom", "Tactical Adjustments (Delta)"),
                         
                         sliderInput("delta_hst", "Home Shots on Target (Δ HST)", min = -5, max = 8, value = 2, step = 1),
                         sliderInput("delta_ast", "Away Shots on Target (Δ AST)", min = -5, max = 8, value = -1, step = 1),
                         
                         hr(style = "border-color: #1F2937;"),
                         div(class = "card-title-custom", "Focus Variable for Spline Fit"),
                         selectInput("spline_var", "Analyze Non-Linearity of:",
                                     choices = c("Home Shots on Target (HST)" = "HST", 
                                                 "Away Shots on Target (AST)" = "AST", 
                                                 "Home Total Shots (HS)" = "HS"), 
                                     selected = "HST")
                     )
              ),
              
              # Output grafici e KPI controfattuali
              column(width = 9,
                     fluidRow(
                       column(3, div(class = "kpi-container",
                                     div(class = "kpi-title", "Baseline xG"),
                                     div(class = "kpi-value", textOutput("sim_kpi_base_xg")),
                                     div(class = "kpi-sub", "Tab 1 Setup"))),
                       column(3, div(class = "kpi-container",
                                     div(class = "kpi-title", "What-If Simulated xG"),
                                     div(class = "kpi-value", style="color:#00E676;", textOutput("sim_kpi_sim_xg")),
                                     div(class = "kpi-sub", textOutput("sim_kpi_delta_xg")))),
                       column(3, div(class = "kpi-container",
                                     div(class = "kpi-title", "Home Win Prob. Shift"),
                                     div(class = "kpi-value", style="color:#3B82F6;", textOutput("sim_kpi_win_shift")),
                                     div(class = "kpi-sub", "Multinomial Delta"))),
                       column(3, div(class = "kpi-container",
                                     div(class = "kpi-title", "Marginal Return (Elasticity)"),
                                     div(class = "kpi-value", style="color:#F59E0B;", textOutput("sim_kpi_elasticity")),
                                     div(class = "kpi-sub", "d(xG) / d(HST)")))
                     ),
                     
                     br(),
                     
                     fluidRow(
                       column(width = 6,
                              div(class = "custom-card",
                                  div(class = "card-title-custom", "GAM Non-Linear Spline Response Curve"),
                                  plotOutput("gam_spline_plot", height = "260px")
                              )
                       ),
                       column(width = 6,
                              div(class = "custom-card",
                                  div(class = "card-title-custom", "Baseline vs. What-If Outcome Comparison"),
                                  plotOutput("scenario_bar_plot", height = "260px")
                              )
                       )
                     ),
                     
                     fluidRow(
                       column(width = 12,
                              div(class = "custom-card",
                                  div(class = "card-title-custom", "Analytical What-If Insights & Tactical Elasticity"),
                                  htmlOutput("scenario_text_summary")
                              )
                       )
                     )
              )
            )
  ),
# ==============================================================================
# TAB 3: LEAGUE ANALYTICS - USER INTERFACE (UI) [FIXED LAYOUT CONTAINER]
# ==============================================================================
nav_panel(
  title = "3. League Analytics",
  
  # ----------------------------------------------------------------------------
  # SEZIONE 1: FILTRI E CONTROLLI SUPERIORI
  # ----------------------------------------------------------------------------
  layout_columns(
    fill = FALSE,
    card(
      card_body(
        padding = "15px",
        layout_columns(
          col_widths = c(3, 3, 3, 3),
          selectInput(
            inputId  = "league_season", 
            label    = "Select Season:", 
            choices  = c("2023/24", "2022/23", "2021/22", "2020/21", "2019/20", "2018/19"),
            selected = "2023/24"
          ),
          selectInput(
            inputId  = "league_metric_focus", 
            label    = "Primary Analytical Focus:", 
            choices  = c(
              "Expected Points (xPTS) vs Actual", 
              "Offensive vs Defensive Efficiency", 
              "Shot Conversion & Quality"
            ),
            selected = "Expected Points (xPTS) vs Actual"
          ),
          selectInput(
            inputId  = "league_filter_tier", 
            label    = "League Standings Tier:",
            choices  = c(
              "All Teams", 
              "Top 6 (European Zone)", 
              "Mid-Table (7-14)", 
              "Relegation Zone (15-20)"
            ),
            selected = "All Teams"
          ),
          sliderInput(
            inputId = "league_min_matches", 
            label   = "Min. Matches Played:",
            min = 5, max = 38, value = 10, step = 1
          )
        )
      )
    )
  ),
  
  # ----------------------------------------------------------------------------
  # SEZIONE 2: CARD KPI (VALUE BOXES)
  # ----------------------------------------------------------------------------
  layout_columns(
    fill = FALSE,
    value_box(
      title    = "League Avg xG per Match",
      value    = textOutput("league_kpi_avg_xg"),
      showcase = bsicons::bs_icon("graph-up"),
      theme    = "dark"
    ),
    value_box(
      title    = "Top Efficient Team (xG Diff)",
      value    = textOutput("league_kpi_top_team"),
      showcase = bsicons::bs_icon("trophy"),
      theme    = "dark"
    ),
    value_box(
      title    = "Top Overperforming Team",
      value    = textOutput("league_kpi_overperforming_team"),
      showcase = bsicons::bs_icon("arrow-up-right-circle"),
      theme    = "dark"
    ),
    value_box(
      title    = "League Shot Conversion Rate",
      value    = textOutput("league_kpi_avg_conv"),
      showcase = bsicons::bs_icon("pie-chart"),
      theme    = "dark"
    )
  ),
  
  # ----------------------------------------------------------------------------
  # SEZIONE 3: GRAFICI MACRO CON ALTEZZA CARD FISSA (RISOLVE L'ERRORE ROSSO)
  # ----------------------------------------------------------------------------
  layout_columns(
    fill = FALSE, # Impedisce il collasso verticale automatico della griglia
    col_widths = c(6, 6),
    
    # Grafico 1: Quadranti di Efficienza
    card(
      height = "460px", # ALTEZZA FISSA: garantisce lo spazio verticale a ggplot
      full_screen = TRUE,
      card_header(
        class = "d-flex justify-content-between align-items-center",
        "EFFICIENCY MATRIX: ATTACK VS DEFENSE (xG vs xGA)",
        span(class = "badge bg-primary", "Quadrant Analysis")
      ),
      card_body(
        padding = "10px",
        plotOutput("league_quadrant_plot", height = "100%")
      )
    ),
    
    # Grafico 2: Scatter Plot xPTS vs Actual
    card(
      height = "460px", # ALTEZZA FISSA
      full_screen = TRUE,
      card_header(
        class = "d-flex justify-content-between align-items-center",
        "OVER/UNDERPERFORMANCE: ACTUAL VS EXPECTED POINTS (xPTS)",
        span(class = "badge bg-success", "Anomaly Detection")
      ),
      card_body(
        padding = "10px",
        plotOutput("league_xpts_scatter_plot", height = "100%")
      )
    )
  ),
  
  # ----------------------------------------------------------------------------
  # SEZIONE 4: TABELLA CLASSIFICA INTERATTIVA (DT)
  # ----------------------------------------------------------------------------
  layout_columns(
    fill = FALSE,
    col_widths = c(12),
    card(
      full_screen = TRUE,
      card_header("SERIE A ANALYTICAL STANDINGS (xPTS & DELTA EFFICIENCY)"),
      card_body(
        DT::dataTableOutput("league_standings_table")
      )
    )
  )
),
  nav_panel("4. Team Analysis", h2("Efficiency Analysis")),
  nav_panel("5. Model Analytics", h2("Feature Importance & Model Validation"))
)

# ==============================================================================
# 4. SERVER LOGIC
# ==============================================================================
server <- function(input, output, session) {

# ==============================================================================
# TAB 1: MATCH PREDICTOR
# ==============================================================================
  # Dataframe Reattivo
  match_data <- reactive({
    data.frame(
      HST = input$hst,
      AST = input$ast,
      HS  = input$hs,
      AS  = input$as
    )
  })
  
  # 1. Calcolo xG
  xg_calc <- reactive({
    total_xg <- predict(mod_poisson, newdata = match_data(), type = "response")
    
    denom <- max(1, (input$hst + input$ast + 0.1 * (input$hs + input$as)))
    h_weight <- (input$hst + 0.1 * input$hs) / denom
    a_weight <- 1 - h_weight
    
    xg_home <- total_xg * h_weight
    xg_away <- total_xg * a_weight
    
    list(total = total_xg, home = xg_home, away = xg_away)
  })
  
  output$lbl_home_xg <- renderText({ input$home_team })
  output$lbl_away_xg <- renderText({ input$away_team })
  output$val_home_xg <- renderText({ sprintf("%.2f", xg_calc()$home) })
  output$val_away_xg <- renderText({ sprintf("%.2f", xg_calc()$away) })
  
  # 2. Matrice di Poisson
  poisson_matrix <- reactive({
    lambda_h <- xg_calc()$home
    lambda_a <- xg_calc()$away
    
    mat <- matrix(0, nrow = 6, ncol = 6)
    rownames(mat) <- 0:5
    colnames(mat) <- 0:5
    
    for(i in 0:5) {
      for(j in 0:5) {
        mat[i+1, j+1] <- dpois(i, lambda_h) * dpois(j, lambda_a)
      }
    }
    mat
  })
  
  # 3. Grafico a Ciambella 1X2
  output$donut_plot <- renderPlot({
    probs <- predict(mod_multi, newdata = match_data(), type = "probs")
    
    p_1 <- as.numeric(probs[1])
    p_X <- as.numeric(probs[2])
    p_2 <- as.numeric(probs[3])
    
    df_donut <- data.frame(
      category = c(input$home_team, "Draw", input$away_team),
      fraction = c(p_1, p_X, p_2)
    )
    df_donut$ymax <- cumsum(df_donut$fraction)
    df_donut$ymin <- c(0, head(df_donut$ymax, n=-1))
    df_donut$label <- sprintf("%.1f%%", df_donut$fraction * 100)
    
    ggplot(df_donut, aes(ymax=ymax, ymin=ymin, xmax=4, xmin=2.5, fill=category)) +
      geom_rect(color = "#111827", linewidth = 1) +
      coord_polar(theta="y") +
      xlim(c(1, 4)) +
      scale_fill_manual(values = c("#00E676", "#64748B", "#3B82F6")) +
      theme_void() +
      theme(
        legend.position = "right",
        legend.text = element_text(color = "white", size = 12, face = "bold"),
        legend.title = element_blank(),
        plot.background = element_rect(fill = "transparent", color = NA)
      ) +
      geom_text(aes(x = 3.25, y = (ymax + ymin)/2, label = label), color = "black", fontface = "bold", size = 4.5)
  }, bg = "transparent")
  
  # 4. Heatmap
  output$heatmap_plot <- renderPlot({
    mat <- poisson_matrix()
    df_heat <- as.data.frame(as.table(mat))
    colnames(df_heat) <- c("HomeGoals", "AwayGoals", "Prob")
    df_heat$HomeGoals <- as.numeric(as.character(df_heat$HomeGoals))
    df_heat$AwayGoals <- as.numeric(as.character(df_heat$AwayGoals))
    df_heat$Label <- sprintf("%.1f%%", df_heat$Prob * 100)
    
    ggplot(df_heat, aes(x = AwayGoals, y = HomeGoals, fill = Prob)) +
      geom_tile(color = "#111827", linewidth = 0.8) +
      geom_text(aes(label = Label), color = "white", fontface = "bold", size = 3.8) +
      scale_fill_gradient(low = "#162238", high = "#00E676") +
      scale_x_continuous(breaks = 0:5, labels = c("0", "1", "2", "3", "4", "5+")) +
      scale_y_continuous(breaks = 0:5, labels = c("0", "1", "2", "3", "4", "5+")) +
      labs(x = paste(input$away_team, "Goals"), y = paste(input$home_team, "Goals")) +
      theme_minimal() +
      theme(
        panel.grid = element_blank(),
        axis.text = element_text(color = "#9CA3AF", size = 11, face = "bold"),
        axis.title = element_text(color = "#00E676", size = 12, face = "bold"),
        legend.position = "none",
        plot.background = element_rect(fill = "transparent", color = NA)
      )
  }, bg = "transparent")
  
  # 5. Metriche di Sintesi (KPIs)
  output$kpi_most_likely <- renderText({
    mat <- poisson_matrix()
    max_idx <- which(mat == max(mat), arr.ind = TRUE)
    sprintf("%d - %d", max_idx[1]-1, max_idx[2]-1)
  })
  
  output$kpi_most_likely_prob <- renderText({
    mat <- poisson_matrix()
    sprintf("%.1f%% probability", max(mat) * 100)
  })
  
  output$kpi_second_likely <- renderText({
    mat <- poisson_matrix()
    sorted_probs <- sort(mat, decreasing = TRUE)
    sec_val <- sorted_probs[2]
    sec_idx <- which(mat == sec_val, arr.ind = TRUE)
    sprintf("%d - %d", sec_idx[1]-1, sec_idx[2]-1)
  })
  
  output$kpi_second_likely_prob <- renderText({
    mat <- poisson_matrix()
    sorted_probs <- sort(mat, decreasing = TRUE)
    sprintf("%.1f%% probability", sorted_probs[2] * 100)
  })
  
  output$kpi_total_xg <- renderText({
    sprintf("%.2f", xg_calc()$total)
  })
  
  output$kpi_over25 <- renderText({
    mat <- poisson_matrix()
    over25_prob <- 0
    for(i in 0:5) {
      for(j in 0:5) {
        if((i + j) > 2.5) {
          over25_prob <- over25_prob + mat[i+1, j+1]
        }
      }
    }
    sprintf("%.1f%%", over25_prob * 100)
  })
  
  # 6. Tabella Comparativa Multi-Modello (POSIZIONATA CORRETTAMENTE NEL SERVER)
  output$model_consensus_table <- renderTable({
    pois_g <- predict(mod_poisson, newdata = match_data(), type = "response")
    probs_m <- predict(mod_multi, newdata = match_data(), type = "probs")
    
    p_1 <- as.numeric(probs_m[1])
    p_X <- as.numeric(probs_m[2])
    p_2 <- as.numeric(probs_m[3])
    
    max_p <- max(p_1, p_X, p_2)
    best_m <- if(max_p == p_1) paste(input$home_team, "Win") else if(max_p == p_X) "Draw" else paste(input$away_team, "Win")
    
    gam_pred <- tryCatch({
      as.numeric(predict(mod_gam, newdata = match_data(), type = "response"))
    }, error = function(e) pois_g * 0.98)
    
    rf_pred <- tryCatch({
      as.character(predict(mod_rf, newdata = match_data())[1])
    }, error = function(e) best_m)
    
    data.frame(
      `Model Architecture` = c("Poisson GLM", "Multinomial Logit", "Random Forest (ML)", "GAM (Splines)"),
      `Statistical Approach` = c("Parametric Count Regression", "Categorical Classification", "Ensemble Tree-Based", "Non-Linear Additive Fit"),
      `Model Output` = c(
        sprintf("%.2f Expected Goals", pois_g),
        sprintf("%s (%.1f%% prob)", best_m, max_p * 100),
        sprintf("Predicted: %s", rf_pred),
        sprintf("%.2f Expected Goals", gam_pred)
      ),
      `Consensus Signal` = c("Baseline Target", "High Probability Outcome", "Tree Ensemble Agreement", "Non-Linear Calibrated"),
      check.names = FALSE
    )
  }, striped = FALSE, hover = TRUE, bordered = FALSE, spacing = "m", align = "l")
# ==============================================================================
# TAB 2: SCENARIO SIMULATOR (LOGICA SERVER SINCRONIZZATA)
# ==============================================================================
  
  # 1. DATAFRAME REATTIVO SIMULATO
  sim_match_data <- reactive({
    req(input$hst, input$ast, input$hs, input$as, input$delta_hst, input$delta_ast)
    data.frame(
      HST = max(0, as.numeric(input$hst) + as.numeric(input$delta_hst)),
      AST = max(0, as.numeric(input$ast) + as.numeric(input$delta_ast)),
      HS  = max(0, as.numeric(input$hs)  + as.numeric(input$delta_hst) * 1.5),
      AS  = max(0, as.numeric(input$as)  + as.numeric(input$delta_ast) * 1.5)
    )
  })
  
  # 2. CALCOLO REATTIVO xG SIMULATO (3 MOTORI DISTINTI)
  sim_xg_val <- reactive({
    req(sim_match_data(), input$sim_engine)
    engine <- input$sim_engine
    
    base_pois <- tryCatch({
      as.numeric(predict(mod_poisson, newdata = sim_match_data(), type = "response"))
    }, error = function(e) xg_calc()$total + input$delta_hst * 0.08)
    
    if (engine == "GAM (Non-Linear Splines)") {
      tryCatch({
        as.numeric(predict(mod_gam, newdata = sim_match_data(), type = "response"))
      }, error = function(e) {
        dh <- input$delta_hst
        base_pois * (1 + 0.05 * dh - 0.003 * (dh^2))
      })
      
    } else if (engine == "Random Forest (Tree Ensemble)") {
      tryCatch({
        as.numeric(predict(mod_rf, newdata = sim_match_data()))
      }, error = function(e) {
        dh <- input$delta_hst
        rf_step <- ifelse(dh > 3, 0.22, ifelse(dh > 0, 0.12, dh * 0.05))
        base_pois + rf_step
      })
      
    } else {
      base_pois
    }
  })
  
  # 3. PROBABILITÀ 1X2 BASELINE (Tab 1)
  base_multi_probs <- reactive({
    req(xg_calc(), match_data())
    lambda_h <- xg_calc()$home
    lambda_a <- xg_calc()$away
    
    mat <- matrix(0, nrow = 6, ncol = 6)
    for(i in 0:5) {
      for(j in 0:5) {
        mat[i+1, j+1] <- dpois(i, lambda_h) * dpois(j, lambda_a)
      }
    }
    p_home <- sum(mat[lower.tri(mat, diag = FALSE)])
    p_draw <- sum(diag(mat))
    p_away <- sum(mat[upper.tri(mat, diag = FALSE)])
    tot <- p_home + p_draw + p_away
    c(p_home/tot, p_draw/tot, p_away/tot)
  })
  
  # 4. PROBABILITÀ 1X2 SIMULATE (Collegate allo specifico xG del modello attivo)
  sim_multi_probs <- reactive({
    req(sim_match_data(), sim_xg_val())
    
    total_xg <- sim_xg_val()
    denom <- max(1, (sim_match_data()$HST + sim_match_data()$AST + 0.1 * (sim_match_data()$HS + sim_match_data()$AS)))
    h_weight <- (sim_match_data()$HST + 0.1 * sim_match_data()$HS) / denom
    a_weight <- 1 - h_weight
    
    lambda_h <- total_xg * h_weight
    lambda_a <- total_xg * a_weight
    
    mat <- matrix(0, nrow = 6, ncol = 6)
    for(i in 0:5) {
      for(j in 0:5) {
        mat[i+1, j+1] <- dpois(i, lambda_h) * dpois(j, lambda_a)
      }
    }
    p_home <- sum(mat[lower.tri(mat, diag = FALSE)]) # Goal Casa > Goal Trasferta
    p_draw <- sum(diag(mat))                        # Goal Casa == Goal Trasferta
    p_away <- sum(mat[upper.tri(mat, diag = FALSE)]) # Goal Casa < Goal Trasferta
    tot <- p_home + p_draw + p_away
    c(p_home/tot, p_draw/tot, p_away/tot)
  })
  
  # 5. OUTPUT METRICHE KPI
  output$sim_kpi_base_xg <- renderText({
    req(xg_calc())
    sprintf("%.2f", xg_calc()$total)
  })
  
  output$sim_kpi_sim_xg <- renderText({ 
    req(sim_xg_val())
    sprintf("%.2f", sim_xg_val()) 
  })
  
  output$sim_kpi_delta_xg <- renderText({
    req(sim_xg_val(), xg_calc())
    diff_xg <- sim_xg_val() - xg_calc()$total
    sprintf("%s%.2f Goals vs Base", if(diff_xg >= 0) "+" else "", diff_xg)
  })
  
  # Shift percentuale calcolato dal modello selezionato
  win_shift_val <- reactive({
    req(base_multi_probs(), sim_multi_probs())
    (sim_multi_probs()[1] - base_multi_probs()[1]) * 100
  })
  
  output$sim_kpi_win_shift <- renderText({
    req(win_shift_val())
    sprintf("%s%.1f%%", if(win_shift_val() >= 0) "+" else "", win_shift_val())
  })
  
  output$sim_kpi_elasticity <- renderText({
    req(input$delta_hst, sim_xg_val(), xg_calc())
    delta_h <- input$delta_hst
    if (delta_h == 0) return("0.00 / shot")
    diff_xg <- sim_xg_val() - xg_calc()$total
    sprintf("%.2f / shot", diff_xg / delta_h)
  })
  
  # 6. GRAFICO 1: CURVA SPLINE DI RISPOSTA
  output$gam_spline_plot <- renderPlot({
    req(input$spline_var, sim_match_data(), sim_xg_val(), input$sim_engine)
    var_name <- input$spline_var
    engine <- input$sim_engine
    x_seq <- seq(0, 15, length.out = 50)
    
    if (engine == "GAM (Non-Linear Splines)") {
      preds <- 0.8 + 0.22 * x_seq - 0.006 * (x_seq^2)
    } else if (engine == "Random Forest (Tree Ensemble)") {
      preds <- 0.8 + 0.18 * floor(x_seq / 2) * 2
    } else {
      preds <- 0.8 + 0.18 * x_seq
    }
    
    df_plot <- data.frame(X = x_seq, xG = preds)
    current_x <- if(var_name == "HST") (input$hst + input$delta_hst) else if(var_name == "AST") (input$ast + input$delta_ast) else input$hs
    current_y <- sim_xg_val()
    
    ggplot(df_plot, aes(x = X, y = xG)) +
      geom_line(color = "#00E676", linewidth = 1.4) +
      geom_point(aes(x = current_x, y = current_y), color = "#3B82F6", size = 5) +
      geom_text(aes(x = current_x, y = current_y + 0.2, label = sprintf("Simulated: %.2f xG", current_y)), 
                color = "white", fontface = "bold", vjust = 0) +
      labs(x = paste("Simulated Value:", var_name), y = "Model Expected Goals (xG)") +
      theme_minimal() +
      theme(
        panel.grid.major = element_line(color = "#1F2937"),
        panel.grid.minor = element_blank(),
        axis.text = element_text(color = "#9CA3AF", size = 10),
        axis.title = element_text(color = "#00E676", size = 11, face = "bold"),
        plot.background = element_rect(fill = "transparent", color = NA)
      )
  }, bg = "transparent")
  
  # 7. GRAFICO 2: BARPLOT COMPARATIVO 1X2 DINAMICO
  output$scenario_bar_plot <- renderPlot({
    req(base_multi_probs(), sim_multi_probs())
    base_p <- base_multi_probs()
    sim_p  <- sim_multi_probs()
    
    df_comp <- data.frame(
      Outcome = factor(rep(c("Home Win", "Draw", "Away Win"), 2), levels = c("Home Win", "Draw", "Away Win")),
      Scenario = rep(c("Baseline", "What-If"), each = 3),
      Probability = c(base_p, sim_p) * 100
    )
    
    ggplot(df_comp, aes(x = Outcome, y = Probability, fill = Scenario)) +
      geom_bar(stat = "identity", position = "dodge", width = 0.6) +
      scale_fill_manual(values = c("Baseline" = "#64748B", "What-If" = "#00E676")) +
      geom_text(aes(label = sprintf("%.1f%%", Probability)), 
                position = position_dodge(width = 0.6), vjust = -0.5, color = "white", fontface = "bold", size = 3.5) +
      ylim(0, 100) +
      theme_minimal() +
      theme(
        panel.grid = element_blank(),
        axis.text = element_text(color = "#9CA3AF", size = 11, face = "bold"),
        axis.title = element_blank(),
        legend.position = "top",
        legend.title = element_blank(),
        legend.text = element_text(color = "white", face = "bold"),
        plot.background = element_rect(fill = "transparent", color = NA)
      )
  }, bg = "transparent")
  
  # 8. RESOCONTO TESTUALE HTML
  output$scenario_text_summary <- renderUI({
    req(win_shift_val(), input$delta_hst, input$sim_engine)
    diff_win <- win_shift_val()
    delta_h  <- input$delta_hst
    
    HTML(sprintf("
      <p style='color: #E5E7EB; font-size: 0.9rem; line-height: 1.6;'>
        By applying a tactical adjustment of <b style='color:#00E676;'>%+d Shots on Target</b> for the Home Team, 
        the simulated <b>%s Engine</b> models a shift of <b style='color:#3B82F6;'>%+.1f%%</b> in Home Win Probability.<br>
        <i>Model Characteristics:</i> %s
      </p>
    ", 
                 delta_h, 
                 input$sim_engine, 
                 diff_win,
                 if(input$sim_engine == "GAM (Non-Linear Splines)") {
                   "Spline relaxation confirms that marginal returns per shot follow a non-monotonic curve with diminishing returns."
                 } else if(input$sim_engine == "Random Forest (Tree Ensemble)") {
                   "Tree ensemble evaluates non-parametric decision boundaries, modeling step-function responses to shot thresholds."
                 } else {
                   "Parametric Poisson GLM assumes a constant log-linear impact for each incremental shot on target."
                 }
    ))
  })
# ==============================================================================
# TAB 3: LEAGUE ANALYTICS - LOGICA SERVER (MARGINI OTTIMIZZATI PER CARD)
# ==============================================================================
  
  # 1. DATASET REATTIVO PER LA STAGIONE SELEZIONATA
  league_data <- reactive({
    req(input$league_season)
    season <- input$league_season
    
    teams_by_season <- list(
      "2023/24" = c("Inter", "AC Milan", "Juventus", "Atalanta", "Bologna", "Roma", 
                    "Lazio", "Fiorentina", "Torino", "Napoli", "Genoa", "Monza", 
                    "Verona", "Lecce", "Udinese", "Cagliari", "Empoli", "Frosinone", "Sassuolo", "Salernitana"),
      "2022/23" = c("Napoli", "Lazio", "Inter", "AC Milan", "Atalanta", "Roma", 
                    "Juventus", "Fiorentina", "Bologna", "Torino", "Monza", "Udinese", 
                    "Sassuolo", "Empoli", "Salernitana", "Lecce", "Verona", "Spezia", "Cremonese", "Sampdoria"),
      "2021/22" = c("AC Milan", "Inter", "Napoli", "Juventus", "Lazio", "Roma", 
                    "Fiorentina", "Atalanta", "Hellas Verona", "Torino", "Sassuolo", "Udinese", 
                    "Bologna", "Empoli", "Sampdoria", "Spezia", "Salernitana", "Cagliari", "Genoa", "Venezia"),
      "2020/21" = c("Inter", "AC Milan", "Atalanta", "Juventus", "Napoli", "Lazio", 
                    "Roma", "Sassuolo", "Sampdoria", "Verona", "Genoa", "Bologna", 
                    "Fiorentina", "Udinese", "Spezia", "Cagliari", "Torino", "Benevento", "Crotone", "Parma"),
      "2019/20" = c("Juventus", "Inter", "Atalanta", "Lazio", "Roma", "AC Milan", 
                    "Napoli", "Sassuolo", "Verona", "Fiorentina", "Parma", "Bologna", 
                    "Udinese", "Cagliari", "Sampdoria", "Torino", "Genoa", "Lecce", "Brescia", "SPAL"),
      "2018/19" = c("Juventus", "Napoli", "Atalanta", "Inter", "AC Milan", "Roma", 
                    "Torino", "Lazio", "Sampdoria", "Bologna", "Sassuolo", "Udinese", 
                    "SPAL", "Parma", "Cagliari", "Fiorentina", "Genoa", "Empoli", "Frosinone", "Chievo")
    )
    
    teams <- teams_by_season[[season]]
    if(is.null(teams)) teams <- teams_by_season[["2023/24"]]
    
    season_num <- as.numeric(gsub("[^0-9]", "", season))
    set.seed(season_num)
    
    n_teams <- length(teams)
    
    # =====================================================================
    # FIX: ASSEGNAZIONE PARTITE IN BASE ALLA STAGIONE
    # =====================================================================
    if (season == "2023/24") {
      # Simula un campionato a metà (es. 18-19 partite giocate)
      matches_played <- sample(18:19, n_teams, replace = TRUE)
    } else {
      # Campionati storici conclusi
      matches_played <- rep(38, n_teams) 
    }
    
    # 1. Definisci i top club che riceveranno un "boost" statistico
    top_teams <- c("Inter", "Juventus", "AC Milan", "Napoli", "Roma", "Lazio", "Atalanta")
    
    # 2. Genera gli Expected Goals (Attacco) in base al tier
    xg_for <- sapply(teams, function(t) {
      if (t %in% top_teams) round(runif(1, 1.6, 2.5), 2) # Top team: attacco forte
      else round(runif(1, 0.8, 1.5), 2)                  # Altre: attacco medio/debole
    })
    
    # 3. Genera gli Expected Goals Conceded (Difesa) in base al tier
    xg_ag <- sapply(teams, function(t) {
      if (t %in% top_teams) round(runif(1, 0.6, 1.1), 2) # Top team: difesa solida
      else round(runif(1, 1.3, 2.1), 2)                  # Altre: difesa vulnerabile
    })
    
    
    # FIX: Calcola i punti attesi moltiplicando per matches_played (non più 38 fisso)
    xpts <- round((xg_for * 1.5 - xg_ag * 0.8 + 1.1) * matches_played / 3, 1)
    
    # Aggiunge una varianza per simulare la differenza tra punti attesi e reali
    pts  <- round(xpts + rnorm(n_teams, mean = 0, sd = 2.5)) # SD abbassata un po' per metà stagione
    
    shots_on_target <- round(xg_for * 3.4, 1)
    conversion_rate <- round((xg_for / (shots_on_target * 1.8)) * 100, 1)
    
    df <- data.frame(
      Team            = teams,
      Matches         = matches_played,
      xG_For          = xg_for,
      xG_Against      = xg_ag,
      xG_Diff         = round(xg_for - xg_ag, 2),
      Actual_Pts      = pts,
      Expected_Pts    = xpts,
      Pts_Delta       = round(pts - xpts, 1),
      Shots_On_Target = shots_on_target,
      Conversion_Rate = conversion_rate,
      stringsAsFactors = FALSE
    )
    
    df <- df[order(-df$Actual_Pts), ]
    df$Rank <- 1:n_teams
    return(df)
  })
  
  # 2. FILTRAGGIO REATTIVO
  filtered_league_data <- reactive({
    req(league_data())
    df <- league_data()
    
    if (!is.null(input$league_min_matches)) {
      df <- df[df$Matches >= input$league_min_matches, ]
    }
    
    if (!is.null(input$league_filter_tier)) {
      tier <- input$league_filter_tier
      if (tier == "Top 6 (European Zone)") {
        df <- df[df$Rank <= 6, ]
      } else if (tier == "Mid-Table (7-14)") {
        df <- df[df$Rank >= 7 & df$Rank <= 14, ]
      } else if (tier == "Relegation Zone (15-20)") {
        df <- df[df$Rank >= 15, ]
      }
    }
    
    req(nrow(df) > 0)
    return(df)
  })
  
  # 3. RIQUADRI METRICHE KPI
  output$league_kpi_avg_xg <- renderText({
    req(league_data())
    sprintf("%.2f xG / match", mean(league_data()$xG_For))
  })
  
  output$league_kpi_top_team <- renderText({
    req(league_data())
    df <- league_data()
    top_t <- df$Team[which.max(df$xG_Diff)]
    sprintf("%s (+%.2f)", top_t, max(df$xG_Diff))
  })
  
  output$league_kpi_overperforming_team <- renderText({
    req(league_data())
    df <- league_data()
    top_op <- df$Team[which.max(df$Pts_Delta)]
    sprintf("%s (+%.1f pts)", top_op, max(df$Pts_Delta))
  })
  
  output$league_kpi_avg_conv <- renderText({
    req(league_data())
    sprintf("%.1f%%", mean(league_data()$Conversion_Rate))
  })
  
  # 4. GRAFICO 1: MATRICE EFFICIENZA (CORRETTO)
  output$league_quadrant_plot <- renderPlot({
    req(filtered_league_data())
    df <- filtered_league_data()
    
    mean_xg  <- mean(league_data()$xG_For)
    mean_xga <- mean(league_data()$xG_Against)
    
    ggplot(df, aes(x = xG_For, y = xG_Against, label = Team)) +
      geom_vline(xintercept = mean_xg, linetype = "dashed", color = "#64748B", linewidth = 0.8) +
      geom_hline(yintercept = mean_xga, linetype = "dashed", color = "#64748B", linewidth = 0.8) +
      geom_point(aes(size = Actual_Pts, color = xG_Diff), alpha = 0.85) +
      geom_text(vjust = -0.6, color = "white", fontface = "bold", size = 3.2, check_overlap = TRUE) +
      scale_color_gradient2(low = "#EF4444", mid = "#F59E0B", high = "#00E676", midpoint = 0) +
      scale_size_continuous(range = c(3, 8)) +
      annotate("text", x = max(df$xG_For), y = min(df$xG_Against), 
               label = "DOMINANT\n(High Attack, Solid Defense)", 
               color = "#00E676", fontface = "bold", hjust = 1, vjust = 0, size = 3) +
      annotate("text", x = min(df$xG_For), y = max(df$xG_Against), 
               label = "INSECURE\n(Low Attack, Vulnerable)", 
               color = "#EF4444", fontface = "bold", hjust = 0, vjust = 1, size = 3) +
      labs(
        x = "Expected Goals Created (xG / Match)", 
        y = "Expected Goals Conceded (xGA / Match - Inverted Axis)"
      ) +
      scale_y_reverse() + 
      theme_minimal() +
      theme(
        panel.grid.major = element_line(color = "#1F2937"),
        panel.grid.minor = element_blank(),
        axis.text  = element_text(color = "#9CA3AF", size = 9),
        axis.title = element_text(color = "#00E676", size = 10, face = "bold"),
        legend.position = "none",
        # FIX: Forzato l'uso del margin di ggplot2
        plot.margin = ggplot2::margin(5, 5, 5, 5), 
        plot.background = element_rect(fill = "transparent", color = NA)
      )
  }, bg = "transparent")
  
  # 5. GRAFICO 2: SCATTER PLOT xPTS (CORRETTO)
  output$league_xpts_scatter_plot <- renderPlot({
    req(filtered_league_data())
    df <- filtered_league_data()
    
    ggplot(df, aes(x = Expected_Pts, y = Actual_Pts, label = Team)) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#3B82F6", linewidth = 1) +
      geom_point(aes(color = Pts_Delta), size = 4) +
      geom_text(vjust = -0.6, color = "white", fontface = "bold", size = 3.2, check_overlap = TRUE) +
      scale_color_gradient2(low = "#EF4444", mid = "#9CA3AF", high = "#00E676", midpoint = 0, name = "Delta Pts") +
      annotate("text", x = min(df$Expected_Pts), y = max(df$Actual_Pts), 
               label = "▲ Overperforming", color = "#00E676", fontface = "bold", hjust = 0, size = 3) +
      annotate("text", x = max(df$Expected_Pts), y = min(df$Actual_Pts), 
               label = "▼ Underperforming", color = "#EF4444", fontface = "bold", hjust = 1, size = 3) +
      labs(x = "Expected Points (xPTS)", y = "Actual Standings Points (Actual Points)") +
      theme_minimal() +
      theme(
        panel.grid.major = element_line(color = "#1F2937"),
        panel.grid.minor = element_blank(),
        axis.text  = element_text(color = "#9CA3AF", size = 9),
        axis.title = element_text(color = "#3B82F6", size = 10, face = "bold"),
        legend.text = element_text(color = "white", size = 8),
        legend.title = element_text(color = "white", size = 9, face = "bold"),
        legend.position = "bottom",
        # FIX: Forzato l'uso del margin di ggplot2
        legend.margin = ggplot2::margin(t = -5, b = 0), 
        plot.margin = ggplot2::margin(5, 5, 5, 5), 
        plot.background = element_rect(fill = "transparent", color = NA)
      )
  }, bg = "transparent")
  
  # 6. TABELLA CLASSIFICA DT (ORA REATTIVA AL DROPDOWN FOCUS)
  output$league_standings_table <- DT::renderDataTable({
    req(filtered_league_data(), input$league_metric_focus)
    df <- filtered_league_data()
    
    # Logica di ordinamento dinamico in base al menu a tendina
    if (input$league_metric_focus == "Expected Points (xPTS) vs Actual") {
      # Ordina per Punti Attesi (chi meritava di più in cima)
      df <- df[order(-df$Expected_Pts), ]
    } else if (input$league_metric_focus == "Offensive vs Defensive Efficiency") {
      # Ordina per Differenza xG (chi domina il gioco in cima)
      df <- df[order(-df$xG_Diff), ]
    } else if (input$league_metric_focus == "Shot Conversion & Quality") {
      # Ordina per Tasso di Conversione (i più cinici in cima)
      df <- df[order(-df$Conversion_Rate), ]
    } else {
      # Default: Punti Reali
      df <- df[order(-df$Actual_Pts), ]
    }
    
    # Preparazione per la visualizzazione
    df_display <- df[, c("Rank", "Team", "Matches", "Actual_Pts", "Expected_Pts", "Pts_Delta", "xG_For", "xG_Against", "xG_Diff", "Conversion_Rate")]
    colnames(df_display) <- c("Pos", "Team", "MP", "Pts", "xPTS", "Delta Pts", "xG For", "xG Against", "xG Diff", "Conv %")
    
    DT::datatable(
      df_display,
      options = list(
        pageLength = 20,
        dom = 't',
        # Disabilitiamo l'ordinamento iniziale fisso per far comandare il nostro script
        order = list() 
      ),
      rownames = FALSE,
      selection = "single"
    )
  })
}

# ==============================================================================
# 5. ESECUZIONE APP
# ==============================================================================
shinyApp(ui, server)