import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import r2_score


# Formatting for DataFrame printing
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

# Load previous analysis from pickle files. 
school_neighbors = pd.read_pickle("school_neighbors.pkl")
modelled_values = pd.read_pickle("combined_rf.pkl")
school_agg_info = pd.read_pickle("school_agg_info.pkl")
school_agg_features = pd.read_pickle("school_agg_features.pkl")
labels_matrix = pd.read_pickle("labels_matrix.pkl")
features_matrix = pd.read_pickle("features_matrix.pkl")

def plot_schools(metric = "College"): 
    """
    Convenience function for generating interactive Altair plot with two components: a map-based visual of high schools and a scatter plot of a given metric. 
    Arguments: 
    metric: "College" or "Graduation"
    """
    
    # Formats labels, tooltips, and initial window size
    if metric == "College": 
        metric_verb = "College"
        x_min = 0
        x_max = 100
    elif metric == "Graduation": 
        metric_verb = "Graduated"
        x_min = 70
        x_max = 100
    else:
        raise Exception(f"Please enter a valid metric. {metric} is not 'College' or 'Graduation'.")
        
    url = "https://raw.githubusercontent.com/deldersveld/topojson/080eb96a46307efd0c4a31f4c11ccabeee5e97dd/countries/us-states/CA-06-california-counties.json"
    counties = alt.topo_feature(url, 'cb_2015_california_county_20m')
    interval = alt.selection_multi()
    individual = alt.selection_single()

    y_min=x_min
    y_max=x_max

    background = alt.Chart(counties).mark_geoshape(
                            fill='blue',
                            stroke='white'
                        ).encode(
                            tooltip=alt.Tooltip("properties.NAME:N", title="County")
                        ).properties(
                                width = 400, 
                                height = 500).project('albersUsa')
    schools = alt.Chart(modelled_values).mark_point(
                    size=80, 
                    stroke="yellow", 
                    fill="yellow", 
                    strokeOpacity=1,
                    fillOpacity=1,
                    shape="diamond"
                ).encode(
                    longitude="Longitude:Q",
                    latitude="Latitude:Q",
                    tooltip=[
                        alt.Tooltip("SchoolName:N", title="School Name"),
                        alt.Tooltip("CohortStudents:Q", title="# Students"), 
                        alt.Tooltip(f"Actual {metric}:Q", title=f"% Students {metric_verb}", format='.3'),
                        alt.Tooltip(f"Predicted {metric}:Q", title=f"% Students {metric_verb} Predicted", format='.3'),
                        alt.Tooltip(f"Student-Weighted Residual {metric}:Q", title=f"Δ Students {metric_verb}", format='.3')]
                ).add_selection(individual
                ).transform_filter(interval)

    rate = alt.Chart(modelled_values).mark_circle(clip=True, color="black"
                    ).encode(
                        x=alt.X(f"Predicted {metric}:Q", title = f"Predicted {metric} Rate [%]", scale=alt.Scale(domain=(x_min,x_max))),
                        y=alt.Y(f"Actual {metric}:Q", title = f"Actual {metric} Rate [%]", scale=alt.Scale(domain=(y_min,y_max))),
                        color=alt.condition(interval, alt.value('black'), alt.value('lightgray')),
                        opacity=alt.condition(individual, alt.value(1.0), alt.value(0.05)),
                        tooltip=[
                            alt.Tooltip("SchoolName:N", title="School Name"),
                            alt.Tooltip("CohortStudents:Q", title="# Students"), 
                            alt.Tooltip(f"Actual {metric}:Q", title=f"% Students {metric_verb}", format='.3'),
                            alt.Tooltip(f"Predicted {metric}:Q", title=f"% Students {metric_verb} Predicted", format='.3')]
                    ).properties(
                        width=500,
                        height=500
                    ).add_selection(interval)

    sd = modelled_values[f"Residual {metric}"].std()
    score = r2_score(modelled_values[f"Actual {metric}"], modelled_values[f"Predicted {metric}"])

    text = alt.Chart({'values':[{}]}).mark_text(
                align="right", baseline="top"
            ).encode(
                x=alt.value(475),  # pixels from left
                y=alt.value(450),  # pixels from top
                text=alt.value(
                        [f"σ = {sd:.2f}", f"R = {score:.4f}"]
                    ),
                size=alt.value(16)
            )

    geographic_panel = background+schools
    scatter_panel = _scatter_background(sd)+rate+text

    return (geographic_panel|scatter_panel).configure_axis(
                                        labelFontSize=14,
                                        titleFontSize=16
                                        ).configure_view(
                                        strokeWidth=0
                                        )

def find_similars(name, num_neighbors = 5, info = "all", filters={}):
    """
    Returns a DataFrame with the information of the school + closest matches to the school. 
    Arguments: 
    name: Name of high school, input as a string. 
    num_neighbors: Number of closest neighbors. Default value is set to 5. 
    info: Type of data included in the returned DataFrame. Valid entries are "demogaphics", "stats", "profiles", or "predictions". 
        "all" -- default value, returns a full DataFrame of all details. 
        "demographics" -- returns information about demographics of school and neighbors
        "stats" -- returns information about the school class sizes, poverty index, free or reduced lunches, and salaries
        "profiles" -- returns same information as both "demographics" and "stats"
        "predictions" -- returns the actual and predicted values for both graduation rate and college preparation rate
    filters: Dictionary of valid categories for filtering. Default value is an empty dictionary. 
        "Magnet": Accepts values of 1 or 0, where 1 is Y and 0 is N. 
        "Charter": Accepts values of 1 or 0, where 1 is Y and 0 is N. 
        "County": Accepts valid strings of county names, e.g., "Los Angeles". 
    """
  
    try: 
        look_up = school_agg_info.query(f"School == '{name}'")[["School", "County", "StatusType"]]
        if look_up.shape[0] == 1: 
            code = look_up.index[0]
        elif look_up.shape[0] == 0:
            raise Exception(f"'{name}' is not a valid school name. Please check for typos and try again.")
        elif look_up.shape[0] > 1: 
            print("There are multiple school entries with that name. Please select the school from the listed values by inputting the index.")
            print(look_up.reset_index())
            val = input()
            code = look_up.index[int(val)]
        schools_nn = pd.merge(modelled_values.loc[school_neighbors.loc[code, "NN_CDSCode"]], 
                     school_agg_features, 
                     how="left", 
                     left_index=True, 
                     right_index=True).round(2)
        schools_nn.insert(1, "County", school_agg_info["County"][schools_nn.index], allow_duplicates=True)
    except: 
        print("You seem to have selected a school that is not in the analyzed dataset. Please select a different school!")
        return None
    
    results = schools_nn.iloc[[0]]
    schools_nn = schools_nn.drop(results.iloc[0].name)
    
    if filters:
        for key in filters.keys():
            if key == "Magnet" or key == "Charter": 
                val = filters[key]
                if val != 1 and val != 0:
                    print(f"An invalid value of {val} was input for the category {key}. This filter will not be used.")
                    continue
            elif key == "County":
                val = f"'{filters[key]}'"
                if not schools_nn["County"].str.contains(filters[key]).any():
                    print(f"An invalid value of {val} was input for the category {key}. This filter will not be used.")
                    continue
                
            else:
                continue
            schools_nn = schools_nn.query(f"{key} == {val}")
        
    results = pd.concat([results, schools_nn.head(num_neighbors)])
    
    base = results.columns[0:2].append(results.columns[13:16])
    predictions = base.append(results.columns[5:13])
    school_demographics = base.append(results.columns[16:33])
    school_stats = base.append(results.columns[33:])
    school_profiles = base.append(results.columns[16:])
    all_info = base.append(results.columns[5:13]).append(results.columns[16:])

    if info == "demographics": 
        return results[school_demographics]
    elif info == "stats":
        return results[school_stats]
    elif info == "profiles":
        return results[school_profiles]
    elif info == "predictions":
        return results[predictions]
    elif info == "all+geography": 
        return results
    else: 
        return results[all_info]
    
def plot_schools_similar(school, num_neighbors = 5, metric = "College", filters={}): 
    """
    Function to generate Altair plot with tooltips and two components: a map-based visual of high schools and a scatter plot of a given metric. Accepts a 
    target school as a string, the number of nearest neighbors to plot alongside, and a metric.  
    Arguments: 
    school: School name as a string.
    num_neighbors: Number of closest neighbors as an int. Default value is set to 5. 
    metric: "College" or "Graduation". Default value is set to "College"
    filters: Dictionary of valid categories for filtering. Default value is an empty dictionary. 
        "Magnet": Accepts values of 1 or 0, where 1 is Y and 0 is N. 
        "Charter": Accepts values of 1 or 0, where 1 is Y and 0 is N. 
        "County": Accepts valid strings of county names, e.g., "Los Angeles". 
    """
    
    # Formats labels, tooltips, and initial window size
    if metric == "College": 
        metric_verb = "College Prep'd"
        x_min = 0
        x_max = 100
    elif metric == "Graduation": 
        metric_verb = "Graduated"
        x_min = 70
        x_max = 100
    else:
        raise Exception(f"Please enter a valid metric. {metric} is not 'College' or 'Graduation'.")
        
    url = "https://raw.githubusercontent.com/deldersveld/topojson/080eb96a46307efd0c4a31f4c11ccabeee5e97dd/countries/us-states/CA-06-california-counties.json"
    counties = alt.topo_feature(url, 'cb_2015_california_county_20m')

    y_min=x_min
    y_max=x_max

    plotted_df = find_similars(school, num_neighbors, info = "all+geography", filters=filters)
    
    background = alt.Chart(counties).mark_geoshape(
                            fill='blue',
                            stroke='white'
                        ).encode(
                            tooltip=alt.Tooltip("properties.NAME:N", title="County")
                        ).properties(
                                width = 400, 
                                height = 500).project('albersUsa')
    schools = alt.Chart(plotted_df).mark_point(
                    size=80, 
                    stroke="yellow", 
                    fill="yellow", 
                    strokeOpacity=1,
                    fillOpacity=1,
                    shape="diamond"
                ).encode(
                    longitude="Longitude:Q",
                    latitude="Latitude:Q",
                    tooltip=[
                        alt.Tooltip("SchoolName:N", title="School Name"),
                        alt.Tooltip("CohortStudents:Q", title="# Students"), 
                        alt.Tooltip(f"Actual {metric}:Q", title=f"% Students {metric_verb}", format='.3'),
                        alt.Tooltip(f"Predicted {metric}:Q", title=f"% Students {metric_verb} Predicted", format='.3'),
                        alt.Tooltip(f"Student-Weighted Residual {metric}:Q", title=f"Δ Students {metric_verb}", format='.3')]
                )

    rate = alt.Chart(plotted_df).mark_circle(clip=True, color="blue", size=100
                    ).encode(
                        x=alt.X(f"Predicted {metric}:Q", title = f"Predicted {metric} Rate [%]", scale=alt.Scale(domain=(x_min,x_max))),
                        y=alt.Y(f"Actual {metric}:Q", title = f"Actual {metric} Rate [%]", scale=alt.Scale(domain=(y_min,y_max))),
                        opacity=alt.value(1.0),
                        tooltip=[
                            alt.Tooltip("SchoolName:N", title="School Name"),
                            alt.Tooltip("CohortStudents:Q", title="# Students"), 
                            alt.Tooltip(f"Actual {metric}:Q", title=f"% Students {metric_verb}", format='.3'),
                            alt.Tooltip(f"Predicted {metric}:Q", title=f"% Students {metric_verb} Predicted", format='.3')]
                    ).properties(
                        width=500,
                        height=500
                    )
    
    rate_all = alt.Chart(modelled_values).mark_circle(clip=True, color="black"
                ).encode(
                    x=alt.X(f"Predicted {metric}:Q", title = f"Predicted {metric} Rate [%]", scale=alt.Scale(domain=(x_min,x_max))),
                    y=alt.Y(f"Actual {metric}:Q", title = f"Actual {metric} Rate [%]", scale=alt.Scale(domain=(y_min,y_max))),
                    opacity=alt.value(0.1),
                    #tooltip=[
                    #    alt.Tooltip("SchoolName:N", title="School Name"),
                    #    alt.Tooltip("CohortStudents:Q", title="# Students"), 
                    #    alt.Tooltip(f"Actual {metric}:Q", title=f"% Students {metric_verb}", format='.3'),
                    #    alt.Tooltip(f"Predicted {metric}:Q", title=f"% Students {metric_verb} Predicted", format='.3')]
                ).properties(
                    width=500,
                    height=500
                )

    sd = modelled_values[f"Residual {metric}"].std()
    score = r2_score(modelled_values[f"Actual {metric}"], modelled_values[f"Predicted {metric}"])

    text = alt.Chart({'values':[{}]}).mark_text(
                align="right", baseline="top"
            ).encode(
                x=alt.value(475),  # pixels from left
                y=alt.value(450),  # pixels from top
                text=alt.value(
                        [f"σ = {sd:.2f}", f"R = {score:.4f}"]
                    ),
                size=alt.value(16)
            )

    geographic_panel = background+schools
    scatter_panel = _scatter_background(sd)+rate_all+rate+text

    return (geographic_panel|scatter_panel).configure_axis(
                                        labelFontSize=14,
                                        titleFontSize=16
                                        ).configure_view(
                                        strokeWidth=0
                                        )

def _scatter_background(offset, xmin=0, xmax=100):
    """
    Internal function to programatically generate background for scatter plots generated with this module. This function may not be maintained
    for backwards compatiability. 
    Arguments: 
    offset: float, usually corresponding to the standard deviation
    xmin: minimum x value of plot, default of 0
    xmax: maximum x value of plot, default of 100
    """
    line_x_min = xmin
    line_x_max = xmax
    line_y_min = line_x_min
    line_y_max = line_x_max
    lines = pd.DataFrame({
        'Actual': [line_x_min,line_x_max],
        'Predicted':  [line_y_min,line_y_max],
        'Predicted_Upper':  [line_y_min+offset, line_y_max+offset],
        'Predicted_Lower':  [line_y_min-offset, line_y_max-offset],
        'Max': [line_y_max, line_y_max]
    })


    center_line = alt.Chart(lines).mark_line(clip=True, color="blue", strokeDash=[9, 3, 9]
                    ).encode(
                        x="Actual", 
                        y="Predicted"
                    )

    lower_line = alt.Chart(lines).mark_line(clip=True, color="red", strokeDash=[9, 3, 9]
                    ).encode(
                        x="Actual", 
                        y="Predicted_Lower"
                    )

    upper_line = alt.Chart(lines).mark_line(clip=True, color="red", strokeDash=[9, 3, 9]
                    ).encode(
                        x="Actual", 
                        y="Predicted_Upper"
                    )

    shaded_area = alt.Chart(lines).mark_area(clip=True, opacity=0.2, color='orange'
                    ).encode(
                        x="Actual", 
                        y="Predicted_Lower",
                        y2="Predicted_Upper"
                    )
    shaded_area_bad = alt.Chart(lines).mark_area(clip=True, opacity=0.2, color='red'
                    ).encode(
                        x="Actual", 
                        y="Predicted_Lower",
                    )
    shaded_area_good = alt.Chart(lines).mark_area(clip=True, opacity=0.2, color='green'
                    ).encode(
                        x="Actual", 
                        y="Max",
                        y2="Predicted_Upper"
                    )
    
    return center_line+upper_line+lower_line+shaded_area+shaded_area_good+shaded_area_bad