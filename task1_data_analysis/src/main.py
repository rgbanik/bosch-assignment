import streamlit as st
from analyzer import DatasetAnalyzer
from visualizer import Visualizer

def main():
    """
    Main method which calls the analyzer and visualizer classes
    and writes to streamlit
    """

    st.write("Hello! Thank you for checking out my assignment submission!")
    st.write("Please wait while the analysis is being generated.")
    st.write("The animated icon to the top right shows that the system is processing the data.")
    st.write("Plots and data will appear here sequentially as they are generated. :D")

    # Initialize dataset analyzer
    dataset_analyzer = DatasetAnalyzer()
    dataset_analyzer.analyze_labels()

    st.write("Although these statistics are useful, they are hard to interpret in tabular form.")
    st.write("Some anomalous statistics can be seen for each class, like very small or very large widths, heights, and areas")
    st.write("However, these might just be outliers and only a few in number. A boxplot could help reveal this.")
    st.write("Let's generate some plots to visualize the data!")

    visualizer = Visualizer()

    visualizer.render_barplots_general()
    visualizer.render_barplots_comparison()
    
    visualizer.generate_boxplot()
    st.write("The boxplots reveal that the train set has several large outliers that cover over 40 percent of the image.")
    st.write("There are two images in the train set with traffic lights that cover almost the entire image.")
    st.write("There are three images with buses and four images with trucks that cover over 70 percent of the images in the train set.")
    st.write("The validation set outliers are much smaller, and there are only about 5 truck images that cover more than 50% of the image.")
    visualizer.generate_heatmaps()
    st.write("We see from these heatmaps that the distribution of cars, traffic lights, and traffic signs are somewhat uniform")
    st.write("and the distribution of bike, motor, and rider are rather haphazard in both sets, despite their small counts.")
    st.write("The distribution of the person class, which has two modes, is an interesting way to learn that the dataset features several scenes where pedestrians are on either side of the road")

    st.write("That brings us to the end of the analysis.")
    st.write("While I would have liked to do a lot more in this task and visualize individual examples, I have unfortunately run out of time.")
    st.write("Thank you!")


if __name__ == "__main__":
    main()
