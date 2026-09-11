#!/usr/bin/env Rscript

# ==============================================================================
# Module: Spatial Transcriptomics Tumor Boundary Extraction & Distance Profiling
# Description: 1. Identifies tumor boundaries using DBSCAN + Concave Hull.
#              2. Computes Euclidean distances from spots/cells to tumor boundaries.
#              3. Categorizes cells into discrete spatial distance belts.
#              4. Generates high-resolution spatial visualization plots.
# ==============================================================================

suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(ggplot2)
  library(dbscan)
  library(concaveman)
})

# ------------------------------------------------------------------------------
# 1. Core Processing Functions
# ------------------------------------------------------------------------------

#' Extract Tumor Boundary Polygons (sf Object)
#'
#' @param data Data frame containing x, y coordinates and cluster column.
#' @param target_clusters Character/numeric vector specifying tumor cluster IDs.
#' @param mode "single" (extract largest primary core) or "multi" (keep distinct sub-clusters).
#' @param eps Density radius for DBSCAN (in spatial coordinates unit, e.g., um).
#' @param min_pts Minimum number of points required to form a DBSCAN core cluster.
#' @param concavity Concavity parameter for concaveman.
extract_tumor_boundaries <- function(data, 
                                     target_clusters = c("0", "1"), 
                                     mode = "multi", 
                                     eps = 100, 
                                     min_pts = 100, 
                                     concavity = 1.2) {
  
  data$clusters <- as.character(data$clusters)
  tumor_points <- data %>% filter(clusters %in% as.character(target_clusters))
  
  if (nrow(tumor_points) == 0) {
    stop("Error: No data points matching target_clusters were found.")
  }
  coords <- as.matrix(tumor_points[, c("x", "y")])
  db_res <- dbscan(coords, eps = eps, minPts = min_pts)
  tumor_points$temp_cluster <- db_res$cluster
  valid_data <- tumor_points %>% filter(temp_cluster != 0)
  if (nrow(valid_data) == 0) {
    message("Warning: No valid spatial clusters found with the given eps/min_pts parameters.")
    return(NULL)
  }
  
  # Target cluster selection mode
  if (mode == "single") {
    target_ids <- valid_data %>%
      group_by(temp_cluster) %>%
      summarise(n = n(), .groups = "drop") %>%
      slice_max(n, n = 1, with_ties = FALSE) %>%
      pull(temp_cluster)
    message(paste("[INFO] Mode: Single Core | Extracted Cluster ID:", target_ids))
    
  } else if (mode == "multi") {
    target_ids <- valid_data %>%
      group_by(temp_cluster) %>%
      summarise(n = n(), .groups = "drop") %>%
      filter(n >= min_pts) %>%
      pull(temp_cluster)
    message(paste("[INFO] Mode: Multi-cluster | Identified Sub-clusters:", length(target_ids)))
  }
  
  # Construct concave hull boundaries per cluster
  boundaries_list <- lapply(target_ids, function(id) {
    cluster_pts <- valid_data %>% filter(temp_cluster == id)
    pts_sf <- st_as_sf(cluster_pts, coords = c("x", "y"))
    hull <- concaveman(pts_sf, concavity = concavity)
    hull$cluster_id <- as.factor(id)
    return(hull)
  })
  
  final_boundaries <- do.call(rbind, boundaries_list)
  return(final_boundaries)
}

#' Calculate Distance to Boundaries and Classify Distance Belts
#'
#' @param df Data frame containing all spatial points.
#' @param boundaries sf polygon object representing tumor boundaries.
#' @param bin_size Step size for distance belts (default: 100um).
#' @param max_dist Maximum cutoff distance for belts (default: 1000um).
calculate_distance_belts <- function(df, boundaries, bin_size = 100, max_dist = 1000) {
  
  message("[INFO] Computing shortest Euclidean distance to tumor boundaries...")
  
  # Convert polygon boundary to MULTILINESTRING for perimeter distance calculation
  tumor_lines <- st_cast(boundaries, "MULTILINESTRING")
  all_points_sf <- st_as_sf(df, coords = c("x", "y"), remove = FALSE)
  
  # Distance matrix computation
  dist_matrix <- st_distance(all_points_sf, tumor_lines)
  df$dist_to_border <- apply(dist_matrix, 1, min)
  
  # Determine if points reside inside boundary polygons
  is_inside <- st_intersects(all_points_sf, boundaries, sparse = FALSE)
  df$is_internal <- apply(is_inside, 1, any)
  
  # Discrete belt bins definition
  breaks <- seq(0, max_dist, by = bin_size)
  labels <- paste0("group", 0:(length(breaks) - 2))
  
  # Assign group tags
  df <- df %>%
    mutate(
      dist_group = case_when(
        dist_to_border <= 5 ~ "Bdy",
        TRUE ~ as.character(cut(dist_to_border, 
                                breaks = breaks, 
                                labels = labels, 
                                include.lowest = FALSE, 
                                right = TRUE))
      )
    ) %>%
    mutate(dist_group = ifelse(is.na(dist_group), "other", dist_group))
  
  # Mark internal points (> 5um from boundary) as "Tumor"
  df$dist_group[df$is_internal & df$dist_group != "Bdy"] <- "Tumor"
  
  # Set factor level ordering
  level_order <- c("Tumor", "Bdy", labels, "other")
  df$dist_group <- factor(df$dist_group, levels = level_order)
  
  return(df)
}

# ------------------------------------------------------------------------------
# 2. Visualization & Export Functions
# ------------------------------------------------------------------------------

plot_spatial_results <- function(df, boundaries, df_final, palette, grid_by = 500) {
  
  x_range <- range(df$x, na.rm = TRUE)
  y_range <- range(df$y, na.rm = TRUE)
  
  base_theme <- theme_minimal() +
    theme(
      panel.background = element_rect(fill = "black", color = NA),
      plot.background  = element_rect(fill = "black", color = NA),
      panel.grid.major = element_line(color = "white", linewidth = 0.2),
      panel.grid.minor = element_line(color = "white", linewidth = 0.1),
      plot.margin      = margin(0, 0, 0, 0, "pt"),
      legend.position  = "none",
      axis.title       = element_blank(),
      axis.text        = element_blank(),
      axis.ticks       = element_blank()
    )
  
  # Plot 1: Tumor Boundary Extraction (BoundarySP)
  p1 <- ggplot() +
    geom_point(data = df %>% filter(clusters %in% c("0","1")), 
               aes(x = x, y = y), color = "grey20", size = 0.05, alpha = 0.2) +
    geom_point(data = df, aes(x = x, y = y, color = as.factor(clusters)), 
               size = 1.2, alpha = 0.8) +
    geom_sf(data = boundaries, fill = NA, color = "white", linewidth = 0.8, linetype = 'dashed') +
    scale_color_manual(values = c("1" = "#FF0000", "2" = "#B29CDC", "3" = "#B1D685", "0" = "grey30")) +
    scale_x_continuous(breaks = seq(floor(x_range[1]), ceiling(x_range[2]), by = grid_by), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(floor(y_range[1]), ceiling(y_range[2]), by = grid_by), expand = c(0, 0)) +
    coord_sf() +
    base_theme
  
  # Plot 2: Distance Belt Profiling (Boundary Both)
  p2 <- ggplot(df_final) +
    geom_point(aes(x = x, y = y, color = dist_group), size = 0.8) +
    scale_color_manual(values = palette, na.value = "#BBDCAD") +
    scale_x_continuous(breaks = seq(floor(x_range[1]), ceiling(x_range[2]), by = grid_by), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq(floor(y_range[1]), ceiling(y_range[2]), by = grid_by), expand = c(0, 0)) +
    coord_sf() +
    base_theme
  
  return(list(boundary_plot = p1, belt_plot = p2))
}

# ------------------------------------------------------------------------------
# 3. Main Pipeline Executor
# ------------------------------------------------------------------------------

run_pipeline <- function(input_csv, output_dir, target_clusters = c("0", "1")) {
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  message("[1/4] Loading input data...")
  df <- read.csv(input_csv)
  
  message("[2/4] Extracting tumor boundary polygons...")
  boundaries <- extract_tumor_boundaries(
    data = df, 
    target_clusters = target_clusters,
    mode = "multi", 
    eps = 100, 
    min_pts = 100, 
    concavity = 1.2
  )
  
  message("[3/4] Calculating spatial distance belts...")
  df_final <- calculate_distance_belts(df, boundaries, bin_size = 100, max_dist = 1000)
  
  # Predefined Color Palette
  my_palette <- c(
    'Tumor'  = '#FFC0CB',
    "Bdy"    = "#d53e4fff", 
    "group0" = "#f46d43ff", "group1" = "#fdae61ff", "group2" = "#fee08b",
    "group3" = "#e6f598",   "group4" = "#abdda4ff", "group5" = "#66c2a5ff",
    "group6" = "#7ad6fb",   "group7" = "#7ebefb",   "group8" = "#b37ad1",
    "group9" = "#8849a7",   "other"  = "#BBDCAD"
  )
  
  message("[4/4] Rendering and exporting spatial figures...")
  plots <- plot_spatial_results(df, boundaries, df_final, palette = my_palette)
  
  ggsave(plot = plots$boundary_plot, 
         filename = file.path(output_dir, "SFig6_Breast_BoundarySP.png"), 
         width = 13.8, height = 6.1, dpi = 300)
  
  ggsave(plot = plots$belt_plot, 
         filename = file.path(output_dir, "SFig6_Breast_Boundary_both.png"), 
         width = 13.8, height = 6.1, dpi = 300)
  
  # Export output metadata CSV containing calculated spatial distance bins
  write.csv(df_final, file.path(output_dir, "Breast_spatial_with_distance.csv"), row.names = FALSE)
  message("[SUCCESS] Pipeline completed successfully. Output saved to: ", output_dir)
}

