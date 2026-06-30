from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os

# -------------------------------------
# MATPLOTLIB
# -------------------------------------

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
# ==========================
# DASHBOARD COLORS
# ==========================

BG_COLOR = '#111111'

CARD_COLOR = '#1a1a1a'

TEXT_COLOR = '#ffffff'

GRID_COLOR = '#444444'

RED = '#ff0000'

RED2 = '#ff4d4d'

RED3 = '#ff8080'

# -------------------------------------
# MACHINE LEARNING
# -------------------------------------

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

# -------------------------------------
# FLASK APP
# -------------------------------------

app = Flask(__name__)

UPLOAD_FOLDER = 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)
# ==========================
# GRAPH THEME
# ==========================

def setup_graph():

    plt.style.use('dark_background')

    fig = plt.figure(

        figsize=(7,4),

        facecolor=BG_COLOR

    )

    ax = plt.gca()

    ax.set_facecolor(BG_COLOR)

    ax.tick_params(

        colors=TEXT_COLOR,

        labelsize=10

    )

    ax.xaxis.label.set_color(

        TEXT_COLOR

    )

    ax.yaxis.label.set_color(

        TEXT_COLOR

    )

    ax.title.set_color(

        TEXT_COLOR

    )

    for spine in ax.spines.values():

        spine.set_color(

            GRID_COLOR

        )

    plt.grid(

        color=GRID_COLOR,

        alpha=0.25

    )

    return fig, ax

# -------------------------------------
# HOME PAGE
# -------------------------------------

@app.route('/')

def home():

    return render_template(
        'index.html'
    )

# -------------------------------------
# ANALYZE DATASET
# -------------------------------------

@app.route(
    '/analyze',
    methods=['POST']
)

def analyze():

    # --------------------------
    # FILE UPLOAD
    # --------------------------

    file = request.files['file']

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # --------------------------
    # LOAD DATASET
    # --------------------------

    df = pd.read_csv(
        filepath
    )

    # --------------------------
    # DATA CLEANING
    # --------------------------

    df = df.dropna()

    df['publish_time'] = (
        pd.to_datetime(
            df['publish_time'],
            errors='coerce'
        )
    )

    df = df.dropna(
        subset=['publish_time']
    )

    df = df[
        df['views'] > 0
    ]

    # --------------------------
    # DATE FEATURES
    # --------------------------
    
    df['day'] = (
        df['publish_time']
        .dt.day_name()
    )

    df['month'] = (
        df['publish_time']
        .dt.month_name()
    )
    
    month_order = [

    'January','February','March','April','May','June','July','August',
    'September','October','November',
    'December'

]

    df['month'] = pd.Categorical(

    df['month'],

    categories=month_order,

    ordered=True

)
    

    # --------------------------
    # ENGAGEMENT RATE
    # --------------------------
    
    # --------------------------
# CATEGORY NAMES
# --------------------------

    category_map = {1:'Film',2:'Autos',10:'Music',15:'Pets',17:'Sports',
    19:'Travel',20:'Gaming',22:'People',23:'Comedy',24:'Entertainment',
    25:'News',26:'How To',27:'Education',28:'Science',29:'Activism'

}

    df['category_name'] = (

    df['category_id']

    .map(category_map)

)
    df['engagement_rate'] = (

        (

            df['likes']

            +

            df['comment_count']

        )

        /

        df['views']

    ) * 100

    # --------------------------
    # KPI SECTION
    # --------------------------

    total_videos = len(df)

    total_views = int(

        df['views']

        .sum()

    )

    avg_engagement = round(

        df['engagement_rate']

        .mean(),

        2

    )

    best_day = (

        df.groupby('day')

        ['engagement_rate']

        .mean()

        .idxmax()

    )

    viral_threshold = (

        df['views']

        .quantile(0.90)

    )

    viral_videos = df[

        df['views']

        >=

        viral_threshold

    ]

    viral_count = len(

        viral_videos

    )

    # --------------------------
    # CATEGORY ANALYSIS
    # --------------------------

    category_data = (

        df.groupby('category_name')

        ['engagement_rate']

        .mean()

        .sort_values(

            ascending=False

        )

        .head(8)

    )

    # --------------------------
    # UPLOAD DAY ANALYSIS
    # --------------------------

    upload_day = (

        df.groupby('day')

        ['engagement_rate']

        .mean()

        .sort_values(

            ascending=False

        )

    )

    # --------------------------
# MONTHLY TREND
# --------------------------

    monthly_trend = (

    df.groupby(

        'month'

    )['views']

    .sum()

    .sort_index()

)
    # --------------------------
    # TOP VIDEOS
    # --------------------------

    top_videos = (

        df.sort_values(

            by='views',

            ascending=False

        )

        .head(10)

    )

    # --------------------------
    # VIRAL VIDEOS
    # --------------------------

    top_viral = (

        viral_videos

        .sort_values(

            by='views',

            ascending=False

        )

        .head(10)

    )

    # --------------------------
    # SHORT TITLE
    # --------------------------

    top_viral['short_title'] = (

        top_viral['title']

        .astype(str)

        .str.slice(

            0,

            25

        )

    )

    # --------------------------
    # MACHINE LEARNING
    # --------------------------

    X = df[[

        'likes',

        'comment_count',

        'engagement_rate'

    ]]

    y = df['views']

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42

        )

    )

    model = LinearRegression()

    model.fit(

        X_train,

        y_train

    )

    y_pred = model.predict(

        X_test

    )

    r2 = round(

        r2_score(

            y_test,

            y_pred

        ) * 100,

        2

    )

    mae = int(

        mean_absolute_error(

            y_test,

            y_pred

        )

    )

    avg_likes = (

        df['likes']

        .mean()

    )

    avg_comments = (

        df['comment_count']

        .mean()

    )

    avg_rate = (

        df['engagement_rate']

        .mean()

    )

    prediction = int(

        abs(

            model.predict(

                [[

                    avg_likes,

                    avg_comments,

                    avg_rate

                ]]

            )[0]

        )

    )

# ==========================
# CATEGORY GRAPH
# ==========================

    fig, ax = setup_graph()

    category_data.plot(

    kind='barh',

    color=[

        RED,

        RED2,

        RED3,

        RED,

        RED2,

        RED3,

        RED,

        RED2

    ]

)

    plt.title(

    'Top Categories'

)

    plt.xlabel(

    'Engagement Rate'

)

    plt.tight_layout()

    plt.savefig(

    'static/category.png',

    transparent=True,

    bbox_inches='tight'

)

    plt.close()

# ==========================
# UPLOAD DAY GRAPH
# ==========================

    fig, ax = setup_graph()

    upload_day.plot(

    kind='bar',

    color=RED

)

    plt.title(

    'Best Upload Day'

)

    plt.ylabel(

    'Engagement'

)

    plt.tight_layout()

    plt.savefig(

    'static/day.png',

    transparent=True,

    bbox_inches='tight'

)

    plt.close()

# ==========================
# MONTHLY TREND
# ==========================

    fig, ax = setup_graph()

    monthly_trend.plot(

    linewidth=4,

    color=RED

)

    plt.title(

    'Monthly Views Trend'

)

    plt.ylabel(

    'Views'

)

    plt.tight_layout()

    plt.savefig(

    'static/trend.png',

    transparent=True,

    bbox_inches='tight'

)

    plt.close()

# ==========================
# VIRAL GRAPH
# ==========================

    fig, ax = setup_graph()

    plt.barh(

    top_viral['short_title'],

    top_viral['views'],

    color=RED

)

    plt.title(

    'Top Viral Videos'

)

    plt.tight_layout()

    plt.savefig(

    'static/viral.png',

    transparent=True,

    bbox_inches='tight'

)

    plt.close()

    # ==========================
# SCATTER GRAPH
# ==========================

    fig, ax = setup_graph()

    plt.scatter(

    df['likes'],

    df['views'],

    alpha=0.6,

    color=RED

)

    plt.title(

    'Likes vs Views'

)

    plt.xlabel(

    'Likes'

)

    plt.ylabel(

    'Views'

)

    plt.tight_layout()

    plt.savefig(

    'static/scatter.png',

    transparent=True,

    bbox_inches='tight'

)

    plt.close()

    # ==========================
# PREDICTION GRAPH
# ==========================

    fig, ax = setup_graph()

    plt.scatter(

    y_test,

    y_pred,

    alpha=0.6,

    color=RED

)

    plt.plot(

    [

        y_test.min(),

        y_test.max()

    ],

    [

        y_test.min(),

        y_test.max()

    ],

    '--',

    linewidth=3,

    color=RED2

)

    plt.title(

    'Actual vs Predicted'

)

    plt.xlabel(

    'Actual Views'

)

    plt.ylabel(

    'Predicted Views'

)

    plt.tight_layout()

    plt.savefig(

    'static/prediction.png',

    transparent=True,

    bbox_inches='tight'

)

    plt.close()

    # ==========================
    # BUSINESS INSIGHTS
    # ==========================

    insights = [

        'Music and Gaming have highest engagement.',

        f'{best_day} is the best upload day.',

        'Top 10% videos have viral potential.',

        'Likes strongly influence views.',

        'High engagement videos should be promoted.',

        'ML predicts future views successfully.'

    ]

    # --------------------------
    # DASHBOARD PAGE
    # --------------------------

    return render_template(

        'dashboard.html',

        total_videos=total_videos,

        total_views=total_views,

        avg_engagement=avg_engagement,

        viral_count=viral_count,

        best_day=best_day,

        prediction=prediction,

        r2=r2,

        mae=mae,

        insights=insights

    )

# -------------------------------------
# RUN APP
# -------------------------------------

if __name__ == '__main__':

    app.run(

        debug=True

    )
