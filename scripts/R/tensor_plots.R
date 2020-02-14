# Script to generate 3-way plots from a PARAFAC analysis.
# usage: Rscript scripts/R/tensor_plots.R --analysis_type vehicle_year_log

library(plyr)
library(dplyr)
library("optparse")

option_list = list(
  make_option("--analysis_type", type="character", default=NULL, 
              help="analysis type; either vehicle_year_log or month_year_log",
              metavar="character"),
  make_option("--unused_out_dir", type="character", default=NULL, 
              help="unused output directory name",
              metavar="character")
)

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

analysis_type = opt$analysis_type

# read data
if (analysis_type == 'vehicle_year_log'){
  sd_lkp = read.csv("./tensor-data/vehicle_year/SystemDescription_vehicle_year_lkp.csv", header = TRUE)
  vid_lkp = read.csv("./tensor-data/vehicle_year/Unit_vehicle_year_lkp.csv", header = TRUE)
  vehicles = read.csv("./raw-data/vehicles.csv", header = TRUE)
  maintenance = read.csv("./raw-data/maintenance.csv", header = TRUE)
  vehicles$Unit. = as.numeric(as.character(vehicles$Unit.))
  temp = left_join(vid_lkp, vehicles, by = c("Unit."))
  A = read.csv("./tensor-data/vehicle_year/A_vehicle_year_log.txt", header = FALSE, row.names = as.character(vid_lkp$Unit.))
  B = read.csv("./tensor-data/vehicle_year/B_vehicle_year_log.txt", header = FALSE, row.names = make.names(sd_lkp$variable))
  C = read.csv("./tensor-data/vehicle_year/C_vehicle_year_log.txt", header = FALSE)
  lambda = read.csv("./tensor-data/vehicle_year/lambda_vehicle_year_log.txt", header = FALSE)
  cluster_membership_matrix = read.csv("./tensor-data/vehicle_year/vehicle_ingroup.txt", sep = " ", header = F)
}

if (analysis_type == 'month_year_log'){
  sd_lkp = read.csv("./tensor-data/month_year/SystemDescription_month_year_lkp.csv", header = TRUE)
  vid_lkp = read.csv("./tensor-data/month_year/Unit_month_year_lkp.csv", header = TRUE)
  vehicles = read.csv("./raw-data/vehicles.csv", header = TRUE)
  maintenance = read.csv("./raw-data/maintenance.csv", header = TRUE)
  vehicles$Unit. = as.numeric(as.character(vehicles$Unit.))
  temp = left_join(vid_lkp, vehicles, by = c("Unit."))
  A = read.csv("./tensor-data/month_year/A_month_year_log.txt", header = FALSE, row.names = as.character(vid_lkp$Unit.))
  B = read.csv("./tensor-data/month_year/B_month_year_log.txt", header = FALSE, row.names = make.names(sd_lkp$variable))
  C = read.csv("./tensor-data/month_year/C_month_year_log.txt", header = FALSE)
  lambda = read.csv("./tensor-data/month_year/lambda_month_year_log.txt", header = FALSE)
  cluster_membership_matrix = read.csv("./tensor-data/month_year/vehicle_ingroup.txt", sep = " ", header = F)
}


library(ggplot2)
library(gridExtra)
library(ggrepel)
library(ggpubr)
A_THRESH = 0.1
B_THRESH = 0.2
OUT_DEVICE = "png" # set to pdf if pdf is preferred
subfolder = analysis_type

## note that maintenance data for month_year starts in 12/2010 (this is first WO open date for these vehicles)
month_year_axis_labs = c("", "2011", rep("", 11), "2012", rep("", 11), "2013", rep("", 11), "2014", rep("", 11), "2015", rep("", 11), "2016", rep("", 11), "2017", rep("", 10))
year_axis_labs = c("0", "1", "2", "3", "4", "5", "6", "7")

for(i in seq(ncol(A))) {
  message(paste0("Generating plot for component ", i))
  
  a_plot = ggplot(A, aes(x = factor(row.names(A)), y = A[,i], fill = temp$Dept.Desc)) + 
    geom_bar(stat = "identity", aes(alpha=factor(cluster_membership_matrix[,i] == 0))) + 
    theme(plot.title = element_text(hjust = 0.5, size = rel(2)), 
          panel.border = element_rect(colour = "black", fill=NA, size=1), 
          panel.background = element_blank(), axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(), axis.title.x = element_blank(), 
          legend.text = element_text(size = rel(0.4)), 
          legend.key.size = unit(0.15, units = "cm"), 
          legend.position=c(0.98,0.15), legend.justification=c(1,0)) + 
    geom_text_repel(aes(label=ifelse(A[,i] == max(A[,i]), gsub("\\.+", ' ', temp$Dept.Desc), '')), nudge_y = 0.05) +
    ylim(0,1) + 
    ggtitle("Vehicle") + 
    labs(fill = "Department") + 
    guides(alpha = FALSE, fill=FALSE) +
    ylab("Factor Weight") +
    scale_alpha_discrete(range=c(1, 0.4))
  b_plot = ggplot(B, aes(x = factor(row.names(B)), y = B[,i])) + geom_bar(stat = "identity") + 
    geom_text_repel(aes(label=ifelse(B[,i] > B_THRESH, gsub("\\.+", ' ', row.names(B)), '')), nudge_y = 0.05, size = 7) + 
    theme(plot.title = element_text(hjust = 0.5, size =rel(2)), 
          panel.border = element_rect(colour = "black", fill=NA, size=1), 
          panel.background = element_blank(), axis.text.x = element_blank(), 
          axis.ticks.x = element_blank(), axis.title.x = element_blank()) + 
    ylim(0,1) + 
    ggtitle("System Description") + 
    ylab("Factor Weight")
  if (analysis_type == 'month_year_log'){
    c_plot = ggplot(C, aes(x = factor(row.names(C)), y = C[,i])) + geom_bar(stat = "identity") + 
      theme(plot.title = element_text(hjust = 0.5, size = rel(2)),
            panel.border = element_rect(colour = "black", fill=NA, size=1),
            panel.background = element_blank(), 
            axis.ticks.x = element_blank(), 
            axis.title.x = element_blank(), 
            axis.text.x = element_text(angle=90, hjust=1, vjust=1)) + 
      ylim(0,1) + 
      ggtitle("Year/Month") + 
      ylab("Factor Weight") + 
      scale_x_discrete(labels = month_year_axis_labs)
  }
  if (analysis_type == 'vehicle_year_log'){
    c_plot = ggplot(C, aes(x = factor(row.names(C)), y = C[,i])) + geom_bar(stat = "identity") + theme(plot.title = element_text(hjust = 0.5, size = rel(2)),panel.border = element_rect(colour = "black", fill=NA, size=1),panel.background = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(), axis.text.x = element_text(angle=90, hjust=1, vjust=1)) + ylim(0,1) + ggtitle("Vehicle Lifetime Year") + ylab("Factor Weight") + scale_x_discrete(labels = year_axis_labs)
  }
  # g = arrangeGrob(a_plot, b_plot, c_plot, nrow = 3, ncol = 1) 
  g = ggarrange(a_plot, b_plot, c_plot, nrow=3, common.legend = TRUE, legend = "bottom")
  outfile = paste0('./img/3_way_plots/', subfolder, '/3_way_plot_factor_', i, '.', OUT_DEVICE)
  ggsave(outfile, g, width = 10.9, height = 6.6, units = "in", device = OUT_DEVICE)
}

