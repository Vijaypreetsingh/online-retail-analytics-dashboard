import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime

st.set_page_config(layout="wide", page_title="Retail Analytics Dashboard")

@st.cache_data
def load_data():
    df = pd.read_excel('Online Retail.xlsx')
    # Data cleaning
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    return df

df = load_data()

# Sidebar filters
st.sidebar.header('Filters')
countries = st.sidebar.multiselect('Country', options=sorted(df['Country'].unique()), default=sorted(df['Country'].unique()))
years = st.sidebar.multiselect('Year', options=sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
months = st.sidebar.multiselect('Month', options=sorted(df['Month'].unique()), default=sorted(df['Month'].unique()))
products = st.sidebar.multiselect('Product', options=sorted(df['Description'].unique()), default=[])
customers = st.sidebar.multiselect('Customer', options=sorted(df['CustomerID'].unique()), default=[])

# Filter data
filtered_df = df[
    (df['Country'].isin(countries)) &
    (df['Year'].isin(years)) &
    (df['Month'].isin(months))
]
if products:
    filtered_df = filtered_df[filtered_df['Description'].isin(products)]
if customers:
    filtered_df = filtered_df[filtered_df['CustomerID'].isin(customers)]

# KPI Section
st.title('Retail Analytics Dashboard')
st.header('Key Performance Indicators')
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric('Total Revenue', f"£{filtered_df['TotalPrice'].sum():,.2f}")
with col2:
    st.metric('Total Orders', filtered_df['InvoiceNo'].nunique())
with col3:
    st.metric('Total Customers', filtered_df['CustomerID'].nunique())
with col4:
    st.metric('Total Products Sold', filtered_df['Quantity'].sum())
with col5:
    avg_order = filtered_df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
    st.metric('Average Order Value', f"£{avg_order:,.2f}")

# Visualizations
st.header('Sales Trend')
sales_trend = filtered_df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
fig = px.line(sales_trend, x='InvoiceDate', y='TotalPrice', title='Revenue Over Time')
st.plotly_chart(fig, use_container_width=True)

st.header('Top Selling Products')
top_products = filtered_df.groupby('Description')['TotalPrice'].sum().nlargest(10).reset_index()
fig = px.bar(top_products, x='TotalPrice', y='Description', orientation='h', title='Top 10 Products by Revenue')
st.plotly_chart(fig, use_container_width=True)

st.header('Country Sales Analysis')
country_sales = filtered_df.groupby('Country')['TotalPrice'].sum().reset_index()
fig = px.bar(country_sales, x='Country', y='TotalPrice', title='Revenue by Country')
st.plotly_chart(fig, use_container_width=True)

st.header('Monthly Revenue')
monthly_sales = filtered_df.groupby(['Year', 'Month'])['TotalPrice'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))
fig = px.line(monthly_sales, x='Date', y='TotalPrice', title='Monthly Sales Trend')
st.plotly_chart(fig, use_container_width=True)

st.header('Customer Analysis')
top_customers = filtered_df.groupby('CustomerID')['TotalPrice'].sum().nlargest(10).reset_index()
fig = px.bar(top_customers, x='TotalPrice', y='CustomerID', orientation='h', title='Top Customers by Spending')
st.plotly_chart(fig, use_container_width=True)

st.header('Product Performance')
freq_products = filtered_df['Description'].value_counts().nlargest(10).reset_index()
freq_products.columns = ['Product', 'Frequency']
fig = px.bar(freq_products, x='Frequency', y='Product', orientation='h', title='Most Frequently Purchased Products')
st.plotly_chart(fig, use_container_width=True)

st.header('Order Distribution')
order_country = filtered_df.groupby('Country')['InvoiceNo'].nunique().reset_index()
fig = px.choropleth(order_country, locations='Country', locationmode='country names', color='InvoiceNo', title='Orders by Country')
st.plotly_chart(fig, use_container_width=True)

st.header('Basket Analysis')
avg_items = filtered_df.groupby('InvoiceNo')['Quantity'].sum().mean()
st.metric('Average Items per Order', f"{avg_items:.2f}")

# RFM Analysis
st.header('Customer Segmentation (RFM Analysis)')
snapshot_date = df['InvoiceDate'].max() + datetime.timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'})

rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'], 4, labels=[1,2,3,4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4])
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

def segment(rfm_score):
    if rfm_score in ['444', '443', '434', '344']:
        return 'High Value'
    elif rfm_score[1] == '4':
        return 'Loyal'
    elif rfm_score[0] == '4':
        return 'At Risk'
    else:
        return 'Others'

rfm['Segment'] = rfm['RFM_Score'].apply(segment)
segment_count = rfm['Segment'].value_counts().reset_index()
segment_count.columns = ['Segment', 'Count']
fig = px.bar(segment_count, x='Segment', y='Count', title='Customer Segments')
st.plotly_chart(fig, use_container_width=True)

# Data Table
st.header('Transaction Data')
st.dataframe(filtered_df)
csv = filtered_df.to_csv(index=False)
st.download_button('Download Filtered Data', csv, 'filtered_data.csv', 'text/csv')