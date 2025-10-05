import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from pydantic import BaseModel
from datetime import datetime, timedelta
import json
import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CATEGORY_KEYWORDS = {
    'Groceries': ['whole foods', 'trader joe', 'safeway', 'kroger', 'grocery', 'target', 'walmart'],
    'Restaurants': ['restaurant', 'cafe', 'coffee', 'doordash', 'uber eats', 'ubereats', 'grubhub'],
    'Transport': ['uber', 'lyft', 'shell', 'chevron', 'exxon', 'parking', 'fuel'],
    'Utilities': ['electric', 'water', 'internet', 'natural gas', 'okc water', 'ong', 'util'],
    'Entertainment': ['netflix', 'spotify', 'hulu', 'amazon prime', 'movie', 'nintendo', 'playstation', 'steam'],
    'Shopping': ['amazon', 'paypal', 'ebay', 'godaddy'],
    'Bills': ['insurance', 'mortgage', 'rent', 'loan'],
    'Other': []
}

# Category hierarchy for multi-layer Sankey
# Format: 'Subcategory': 'Parent Category'
CATEGORY_HIERARCHY = {
    # Bills subcategories
    'Rent': 'Bills',
    'Mortgage': 'Bills',
    'Insurance': 'Bills',
    'Phone': 'Bills',
    'Internet': 'Bills',
    'Cable/TV': 'Bills',
    'Car Payment': 'Bills',
    'Student Loan': 'Bills',
    'Credit Card Payment': 'Bills',

    # Insurance subcategories (deeper level)
    'Health Insurance': 'Insurance',
    'Car Insurance': 'Insurance',
    'Home Insurance': 'Insurance',
    'Life Insurance': 'Insurance',

    # Transport subcategories
    'Gas': 'Transport',
    'Uber/Lyft': 'Transport',
    'Parking': 'Transport',
    'Car Maintenance': 'Transport',
    'Public Transit': 'Transport',
    'Tolls': 'Transport',

    # Kids subcategories
    'Kids Activities': 'Kids',
    'Kids Education': 'Kids',
    'Kids Clothes': 'Kids',
    'Childcare': 'Kids',
    'Kids Sports': 'Kids',

    # Shopping subcategories
    'Online Shopping': 'Shopping',
    'Home Goods': 'Shopping',
    'Electronics': 'Shopping',
    'Clothing': 'Shopping',
    'Books': 'Shopping',

    # Groceries subcategories
    'Supermarket': 'Groceries',
    'Specialty Foods': 'Groceries',
    'Farmers Market': 'Groceries',

    # Restaurants subcategories
    'Fast Food': 'Restaurants',
    'Delivery': 'Restaurants',
    'Coffee Shops': 'Restaurants',
    'Fine Dining': 'Restaurants',

    # Utilities subcategories
    'Electric': 'Utilities',
    'Gas/Heat': 'Utilities',
    'Water': 'Utilities',
    'Trash': 'Utilities',
    'Internet Service': 'Utilities',

    # Entertainment subcategories
    'Streaming': 'Entertainment',
    'Gaming': 'Entertainment',
    'Movies/Theater': 'Entertainment',
    'Concerts/Events': 'Entertainment',
    'Hobbies': 'Entertainment',
}

# Valid categories list (top-level + subcategories)
ALL_CATEGORIES = list(CATEGORY_KEYWORDS.keys()) + list(CATEGORY_HIERARCHY.keys())

CATEGORY_COLORS = {
    'Income': '#2E86AB',  # Blue
    'Groceries': '#06A77D',  # Green
    'Restaurants': '#D62246',  # Red
    'Transport': '#F77F00',  # Orange
    'Utilities': '#8338EC',  # Purple
    'Entertainment': '#FF006E',  # Pink
    'Shopping': '#FFBE0B',  # Yellow
    'Bills': '#3A86FF',  # Light Blue
    'Insurance': '#5A67D8',  # Indigo (for when it's a top-level category)
    'Kids': '#FFB627',  # Orange-Yellow
    'Other': '#8D99AE'  # Gray
}

# Default color for custom categories
DEFAULT_CUSTOM_COLOR = '#A8DADC'

# Database setup
DB_PATH = os.path.join(os.path.expanduser('.'), 'efb_categories.db')

def init_db():
    """Initialize SQLite database for storing custom categories and modifications."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table for custom categories
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS custom_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    ''')

    # Table for transaction category modifications
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS category_modifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def load_custom_categories():
    """Load custom categories from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM custom_categories ORDER BY name')
    categories = [row[0] for row in cursor.fetchall()]
    conn.close()
    return categories

def save_custom_category(name):
    """Save a new custom category to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO custom_categories (name) VALUES (?)', (name,))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def delete_custom_category(name):
    """Delete a custom category from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM custom_categories WHERE name = ?', (name,))
    conn.commit()
    conn.close()

def load_category_modifications():
    """Load all category modifications from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT description, category FROM category_modifications')
    modifications = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return modifications

def save_category_modification(description, category):
    """Save a transaction category modification to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO category_modifications (description, category)
        VALUES (?, ?)
    ''', (description, category))
    conn.commit()
    conn.close()

class TransactionCategorization(BaseModel):
    model_config = {"extra": "forbid"}
    description: str
    category: str
    confidence: float

class BatchCategorizationResult(BaseModel):
    model_config = {"extra": "forbid"}
    transactions: list[TransactionCategorization]

def categorize_transaction(description):
    description = str(description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in description for keyword in keywords):
            return category
    return 'Other'

def validate_category(category):
    """Validate and map GPT-5 category to a valid category."""
    # Direct match
    if category in ALL_CATEGORIES:
        return category

    # Case-insensitive match
    category_lower = category.lower()
    for valid_cat in ALL_CATEGORIES:
        if valid_cat.lower() == category_lower:
            return valid_cat

    # Partial matching for common variations
    mappings = {
        'concert': 'Concerts/Events',
        'concerts': 'Concerts/Events',
        'movie': 'Movies/Theater',
        'movies': 'Movies/Theater',
        'book': 'Books',
        'cloth': 'Clothing',
        'clothes': 'Clothing',
        'gas': 'Gas',
        'fuel': 'Gas',
        'electric': 'Electric',
        'electricity': 'Electric',
    }

    for key, valid_cat in mappings.items():
        if key in category_lower:
            return valid_cat

    # If no match found, return Other
    return 'Other'

def categorize_with_gpt5_streaming(descriptions_list, api_key, update_callback=None, expenses_df=None):
    """Use GPT-5 with streaming output and chunked processing for reliability."""
    if not descriptions_list:
        return {}

    # Determine chunk size - aim for 20-50 transactions per chunk for streaming
    total = len(descriptions_list)
    if total <= 50:
        chunk_size = total  # Don't chunk if 50 or fewer
    else:
        chunk_size = max(20, min(50, total // 20))  # 20-50 transactions per chunk

    client = OpenAI(api_key=api_key)
    categories_str = ', '.join([cat for cat in ALL_CATEGORIES if cat != 'Other'])

    # Prepare schema once
    schema = BatchCategorizationResult.model_json_schema()

    def add_no_additional_properties(obj):
        if isinstance(obj, dict):
            if obj.get('type') == 'object':
                obj['additionalProperties'] = False
            for value in obj.values():
                add_no_additional_properties(value)
        elif isinstance(obj, list):
            for item in obj:
                add_no_additional_properties(item)

    add_no_additional_properties(schema)

    # Split into chunks
    chunks = [descriptions_list[i:i + chunk_size] for i in range(0, len(descriptions_list), chunk_size)]

    # Create expandable section for live categorization
    with st.expander("🔴 Live Categorization Stream", expanded=True):
        status_placeholder = st.empty()
        categorization_list = st.empty()
        progress_bar = st.progress(0)

    all_results = {}
    update_counter = 0

    for chunk_idx, chunk in enumerate(chunks):
        transactions_text = '\n'.join([f"{i+1}. {desc}" for i, desc in enumerate(chunk)])

        # Show chunk progress
        status_placeholder.info(f"⏳ Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} transactions)...")

        try:
            stream = client.responses.create(
                model="gpt-5",
                instructions=f"You are a financial transaction categorizer. Categorize each transaction into one of these categories: {categories_str}. Provide a confidence score between 0 and 1 for each.",
                input=f"Categorize these {len(chunk)} transactions:\n\n{transactions_text}",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "batch_categorization",
                        "schema": schema,
                        "strict": True
                    }
                },
                stream=True
            )

            full_text = ""
            chunk_results = {}

            for event in stream:
                # Handle streaming events based on official docs
                if hasattr(event, 'type'):
                    if event.type == 'response.output_text.delta':
                        # Incremental text chunks
                        if hasattr(event, 'delta'):
                            full_text += event.delta

                            # Update status with progress
                            progress_msg = f"🤖 Chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk_results)}/{len(chunk)} categorized ({len(full_text)} chars)"
                            status_placeholder.info(progress_msg)

                            # Try to parse partial JSON to show progress
                            try:
                                # Attempt to parse what we have so far
                                partial_json = json.loads(full_text)
                                if 'transactions' in partial_json:
                                    # Check for new transactions that were just added
                                    new_transactions = {}
                                    for t in partial_json['transactions']:
                                        desc = t['description']
                                        cat = validate_category(t['category'])  # Validate category
                                        if desc not in chunk_results:
                                            # This is a new categorization
                                            new_transactions[desc] = cat
                                            chunk_results[desc] = cat

                                            # Save to database immediately
                                            save_category_modification(desc, cat)

                                    # Show categorized transactions if we have new ones
                                    if new_transactions:
                                        all_results.update(new_transactions)
                                        cat_text = "\n".join([f"✓ {desc[:50]}... → **{cat}**"
                                                             for desc, cat in list(all_results.items())[-10:]])
                                        categorization_list.markdown(f"**Recent categorizations ({len(all_results)}/{len(descriptions_list)}):**\n\n{cat_text}")

                                        # Update expenses dataframe with new categorizations
                                        if update_callback and expenses_df is not None:
                                            for desc, cat in new_transactions.items():
                                                expenses_df.loc[expenses_df['Description'] == desc, 'Category'] = cat
                                            update_counter += 1
                                            update_callback(expenses_df.copy(), stream_update=update_counter)
                            except json.JSONDecodeError:
                                # Not valid JSON yet, keep accumulating
                                pass

                    elif event.type == 'response.output_text.done':
                        # Final complete text
                        if hasattr(event, 'text'):
                            full_text = event.text

            # Parse final results for this chunk
            output_json = json.loads(full_text)
            chunk_results = {t['description']: validate_category(t['category']) for t in output_json['transactions']}
            all_results.update(chunk_results)

            # Update progress bar
            progress_bar.progress((chunk_idx + 1) / len(chunks))

        except Exception as e:
            st.warning(f"Failed to categorize chunk {chunk_idx + 1}: {str(e)}")
            for desc in chunk:
                all_results[desc] = 'Other'

    # Clean up UI
    progress_bar.empty()
    status_placeholder.success(f"✅ Categorized {len(all_results)} transactions")

    # Fill in any missing descriptions
    for desc in descriptions_list:
        if desc not in all_results:
            all_results[desc] = 'Other'

    return all_results

def categorize_with_gpt5_chunked(descriptions_list, api_key, update_callback=None, expenses_df=None):
    """Use GPT-5 with chunked requests (multiple smaller batches)."""
    if not descriptions_list:
        return {}

    # Determine chunk size based on total transactions
    total = len(descriptions_list)
    if total <= 10:
        chunk_size = max(1, total)  # Don't chunk if 10 or fewer
    else:
        # Aim for 10-20 requests max
        chunk_size = max(1, total // 15)

    client = OpenAI(api_key=api_key)
    categories_str = ', '.join([cat for cat in ALL_CATEGORIES if cat != 'Other'])

    # Prepare schema
    schema = BatchCategorizationResult.model_json_schema()

    def add_no_additional_properties(obj):
        if isinstance(obj, dict):
            if obj.get('type') == 'object':
                obj['additionalProperties'] = False
            for value in obj.values():
                add_no_additional_properties(value)
        elif isinstance(obj, list):
            for item in obj:
                add_no_additional_properties(item)

    add_no_additional_properties(schema)

    # Split into chunks
    chunks = [descriptions_list[i:i + chunk_size] for i in range(0, len(descriptions_list), chunk_size)]

    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, chunk in enumerate(chunks):
        status_text.info(f"🤖 Processing chunk {idx + 1}/{len(chunks)} ({len(chunk)} transactions)...")

        transactions_text = '\n'.join([f"{i+1}. {desc}" for i, desc in enumerate(chunk)])

        try:
            response = client.responses.create(
                model="gpt-5",
                instructions=f"You are a financial transaction categorizer. Categorize each transaction into one of these categories: {categories_str}. Provide a confidence score between 0 and 1 for each.",
                input=f"Categorize these {len(chunk)} transactions:\n\n{transactions_text}",
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "batch_categorization",
                        "schema": schema,
                        "strict": True
                    }
                }
            )

            output_json = json.loads(response.output_text)
            chunk_results = {t['description']: validate_category(t['category']) for t in output_json['transactions']}
            results.update(chunk_results)

            # Update display after each chunk if callback provided
            if update_callback and expenses_df is not None:
                # Apply results so far to expenses
                for desc, cat in results.items():
                    expenses_df.loc[expenses_df['Description'] == desc, 'Category'] = cat
                update_callback(expenses_df.copy())

        except Exception as e:
            st.warning(f"Failed to categorize chunk {idx + 1}: {str(e)}")
            for desc in chunk:
                results[desc] = 'Other'

        progress_bar.progress((idx + 1) / len(chunks))

    progress_bar.empty()
    status_text.empty()

    # Fill in any missing descriptions
    for desc in descriptions_list:
        if desc not in results:
            results[desc] = 'Other'

    return results

def filter_by_date(df, date_filter, start_date=None, end_date=None):
    """Filter dataframe by date range."""
    # Parse dates from the dataframe
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Transaction Date'] if 'Transaction Date' in df.columns else df['Posting Date'], errors='coerce')

    # Remove rows with invalid dates
    df = df.dropna(subset=['Date'])

    if df.empty:
        return df

    today = datetime.now()

    if date_filter == 'Last 7 days':
        cutoff = today - timedelta(days=7)
        df = df[df['Date'] >= cutoff]
    elif date_filter == 'Last 14 days':
        cutoff = today - timedelta(days=14)
        df = df[df['Date'] >= cutoff]
    elif date_filter == 'Last 30 days':
        cutoff = today - timedelta(days=30)
        df = df[df['Date'] >= cutoff]
    elif date_filter == 'Last 60 days':
        cutoff = today - timedelta(days=60)
        df = df[df['Date'] >= cutoff]
    elif date_filter == 'Last 90 days':
        cutoff = today - timedelta(days=90)
        df = df[df['Date'] >= cutoff]
    elif date_filter == 'Last year':
        cutoff = today - timedelta(days=365)
        df = df[df['Date'] >= cutoff]
    elif date_filter == 'Custom range' and start_date and end_date:
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    # 'All time' - no filtering needed

    return df

def prepare_sankey_data(df, use_gpt5=False, api_key=None, display_callback=None, update_callback=None):
    working_df = df.copy()
    working_df['Amount'] = pd.to_numeric(working_df['Amount'], errors='coerce')

    expenses = working_df[working_df['Amount'] < 0].copy()
    expenses['Amount'] = expenses['Amount'].abs()
    expenses['Category'] = expenses['Description'].apply(categorize_transaction)

    def create_sankey_data(expenses_df):
        """Helper to create multi-layer sankey data from expenses dataframe."""
        def hex_to_rgba(hex_color, alpha=0.4):
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r},{g},{b},{alpha})'

        def get_ancestry_chain(category):
            """Get the full chain of parents for a category (from root to category)."""
            chain = [category]
            current = category
            while current in CATEGORY_HIERARCHY:
                parent = CATEGORY_HIERARCHY[current]
                chain.insert(0, parent)
                current = parent
            return chain

        def get_root_category(category):
            """Get the topmost parent category."""
            chain = get_ancestry_chain(category)
            return chain[0]

        # Build multi-layer structure
        labels = ['Income']
        sources = []
        targets = []
        values = []
        node_colors = [CATEGORY_COLORS.get('Income', DEFAULT_CUSTOM_COLOR)]
        link_colors = []
        category_index = {}

        # Group expenses by category and build hierarchy
        category_amounts = expenses_df.groupby('Category')['Amount'].sum().to_dict()

        # Process each category and build the tree structure
        for category, amount in sorted(category_amounts.items(), key=lambda x: x[1], reverse=True):
            chain = get_ancestry_chain(category)

            # Ensure all nodes in the chain exist
            for i, node in enumerate(chain):
                if node not in category_index:
                    labels.append(node)
                    category_index[node] = len(labels) - 1

                    # Determine node color (use root category color)
                    root = chain[0]
                    node_colors.append(CATEGORY_COLORS.get(root, DEFAULT_CUSTOM_COLOR))

            # Create links for the entire chain from Income through all intermediate nodes
            # This ensures the flow is visible at every level
            for i in range(len(chain)):
                if i == 0:
                    # First link: Income -> Root category
                    parent_idx = 0  # Income
                    child_idx = category_index[chain[0]]
                else:
                    # Subsequent links: Parent -> Child
                    parent_idx = category_index[chain[i-1]]
                    child_idx = category_index[chain[i]]

                # Check if this link already exists, if so, add to its value
                link_exists = False
                for j, (src, tgt) in enumerate(zip(sources, targets)):
                    if src == parent_idx and tgt == child_idx:
                        values[j] += amount
                        link_exists = True
                        break

                if not link_exists:
                    sources.append(parent_idx)
                    targets.append(child_idx)
                    values.append(amount)
                    root = get_root_category(category)
                    link_colors.append(hex_to_rgba(CATEGORY_COLORS.get(root, DEFAULT_CUSTOM_COLOR)))

        # Calculate totals for display
        category_totals = (
            expenses_df.groupby('Category', as_index=False)['Amount']
            .sum()
            .sort_values('Amount', ascending=False)
            .reset_index(drop=True)
        )

        sankey_data = {
            'labels': labels,
            'sources': sources,
            'targets': targets,
            'values': values,
            'node_colors': node_colors,
            'link_colors': link_colors
        }

        return category_totals, sankey_data

    # Show initial diagram with known categories
    if display_callback:
        initial_totals, initial_sankey = create_sankey_data(expenses)
        display_callback(initial_totals, initial_sankey, expenses)

    # Use GPT-5 to categorize "Other" transactions
    if use_gpt5 and api_key:
        other_transactions = expenses[expenses['Category'] == 'Other']['Description'].unique().tolist()
        if other_transactions:
            # Use streaming mode with automatic chunking
            def stream_update_callback(updated_expenses, stream_update=None):
                """Called during streaming to update the display."""
                if update_callback:
                    totals, sankey = create_sankey_data(updated_expenses)
                    update_callback(totals, sankey, updated_expenses, stream_update=stream_update)

            gpt_categories = categorize_with_gpt5_streaming(
                other_transactions,
                api_key,
                update_callback=stream_update_callback,
                expenses_df=expenses
            )
            # Update categories based on GPT-5 results
            expenses['Category'] = expenses.apply(
                lambda row: gpt_categories.get(row['Description'], row['Category'])
                if row['Category'] == 'Other' else row['Category'],
                axis=1
            )

    category_totals, sankey_data = create_sankey_data(expenses)
    return category_totals, sankey_data, expenses


def main():
    st.title("💰 Chase Statement Sankey Diagram")

    # Initialize database
    init_db()

    # GPT-5 API Key
    api_key = os.environ.get('OPENAI_API_KEY')

    # Load custom categories from database
    if 'custom_categories' not in st.session_state:
        st.session_state.custom_categories = load_custom_categories()

    # Sidebar: Manage custom categories
    st.sidebar.header("Custom Categories")
    with st.sidebar.expander("Add New Category"):
        new_category_name = st.text_input("Category Name", key="new_cat_name")
        if st.button("Add Category") and new_category_name:
            if save_custom_category(new_category_name):
                st.session_state.custom_categories = load_custom_categories()
                st.success(f"Added category: {new_category_name}")
                st.rerun()
            else:
                st.warning("Category already exists")

    if st.session_state.custom_categories:
        st.sidebar.write("**Your custom categories:**")
        for cat in st.session_state.custom_categories:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.write(f"• {cat}")
            with col2:
                if st.button("×", key=f"del_{cat}"):
                    delete_custom_category(cat)
                    st.session_state.custom_categories = load_custom_categories()
                    st.rerun()

    # Date filtering
    st.sidebar.header("Date Filter")
    date_filter = st.sidebar.selectbox(
        "Select date range",
        ['All time', 'Last 7 days', 'Last 14 days', 'Last 30 days', 'Last 60 days', 'Last 90 days', 'Last year', 'Custom range'],
        index=0
    )

    start_date = None
    end_date = None
    if date_filter == 'Custom range':
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start date")
        with col2:
            end_date = st.date_input("End date")

    uploaded_file = st.file_uploader("Upload Chase CSV", type=['csv'])

    if not uploaded_file:
        return

    # Check if we need to process a new file
    file_hash = hash(uploaded_file.getvalue())
    if 'file_hash' not in st.session_state or st.session_state.file_hash != file_hash:
        # New file uploaded - process it
        st.session_state.file_hash = file_hash

        # Keep the first CSV column as data instead of treating it as an index
        df = pd.read_csv(uploaded_file, index_col=False)

        # Parse dates for the entire dataset
        df['Date'] = pd.to_datetime(df['Transaction Date'] if 'Transaction Date' in df.columns else df['Posting Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        st.session_state.full_df = df
        st.session_state.processed_expenses = None  # Will be set after GPT-5 processing
    else:
        # Use cached data
        df = st.session_state.full_df

    if df.empty:
        st.error("No valid transactions found in CSV.")
        return

    # Load category modifications from database
    if 'category_modifications' not in st.session_state:
        st.session_state.category_modifications = load_category_modifications()

    # Helper function to build sankey data from expenses (consolidated logic)
    def create_sankey_from_expenses(expenses_df):
        """Create multi-layer sankey data from already-categorized expenses."""
        def hex_to_rgba(hex_color, alpha=0.4):
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r},{g},{b},{alpha})'

        def get_ancestry_chain(category):
            """Get the full chain of parents for a category (from root to category)."""
            chain = [category]
            current = category
            while current in CATEGORY_HIERARCHY:
                parent = CATEGORY_HIERARCHY[current]
                chain.insert(0, parent)
                current = parent
            return chain

        def get_root_category(category):
            """Get the topmost parent category."""
            chain = get_ancestry_chain(category)
            return chain[0]

        # Build multi-layer structure
        labels = ['Income']
        sources = []
        targets = []
        values = []
        node_colors = [CATEGORY_COLORS.get('Income', DEFAULT_CUSTOM_COLOR)]
        link_colors = []
        category_index = {}

        # Group expenses by category and build hierarchy
        category_amounts = expenses_df.groupby('Category')['Amount'].sum().to_dict()

        # Process each category and build the tree structure
        for category, amount in sorted(category_amounts.items(), key=lambda x: x[1], reverse=True):
            chain = get_ancestry_chain(category)

            # Ensure all nodes in the chain exist
            for i, node in enumerate(chain):
                if node not in category_index:
                    labels.append(node)
                    category_index[node] = len(labels) - 1

                    # Determine node color (use root category color)
                    root = chain[0]
                    node_colors.append(CATEGORY_COLORS.get(root, DEFAULT_CUSTOM_COLOR))

            # Create links for the entire chain from Income through all intermediate nodes
            # This ensures the flow is visible at every level
            for i in range(len(chain)):
                if i == 0:
                    # First link: Income -> Root category
                    parent_idx = 0  # Income
                    child_idx = category_index[chain[0]]
                else:
                    # Subsequent links: Parent -> Child
                    parent_idx = category_index[chain[i-1]]
                    child_idx = category_index[chain[i]]

                # Check if this link already exists, if so, add to its value
                link_exists = False
                for j, (src, tgt) in enumerate(zip(sources, targets)):
                    if src == parent_idx and tgt == child_idx:
                        values[j] += amount
                        link_exists = True
                        break

                if not link_exists:
                    sources.append(parent_idx)
                    targets.append(child_idx)
                    values.append(amount)
                    root = get_root_category(category)
                    link_colors.append(hex_to_rgba(CATEGORY_COLORS.get(root, DEFAULT_CUSTOM_COLOR)))

        # Calculate totals for display
        category_totals = (
            expenses_df.groupby('Category', as_index=False)['Amount']
            .sum()
            .sort_values('Amount', ascending=False)
            .reset_index(drop=True)
        )

        sankey_data = {
            'labels': labels,
            'sources': sources,
            'targets': targets,
            'values': values,
            'node_colors': node_colors,
            'link_colors': link_colors
        }

        return category_totals, sankey_data

    def display_sankey(category_totals, sankey_data, expenses, stream_update=None):
        """Display or update the Sankey diagram."""
        # Apply any existing modifications and rebuild
        if st.session_state.category_modifications:
            for desc, new_cat in st.session_state.category_modifications.items():
                expenses.loc[expenses['Description'] == desc, 'Category'] = new_cat

            # Rebuild using consolidated function
            category_totals, sankey_data = create_sankey_from_expenses(expenses)

        # Add dollar values to labels
        labels_with_values = [sankey_data['labels'][0]]  # Income node
        for i, label in enumerate(sankey_data['labels'][1:], 1):
            # Find total value for this node by summing all links targeting it
            node_value = sum(val for _src, tgt, val in zip(sankey_data['sources'], sankey_data['targets'], sankey_data['values']) if tgt == i)
            if node_value == 0:  # If it's only a source, use outgoing value
                node_value = sum(val for src, _tgt, val in zip(sankey_data['sources'], sankey_data['targets'], sankey_data['values']) if src == i)
            labels_with_values.append(f"{label}<br>${node_value:,.0f}")

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        label=labels_with_values,
                        color=sankey_data['node_colors']
                    ),
                    link=dict(
                        source=sankey_data['sources'],
                        target=sankey_data['targets'],
                        value=sankey_data['values'],
                        color=sankey_data['link_colors'],
                        label=[f"${v:,.0f}" for v in sankey_data['values']]
                    )
                )
            ]
        )

        fig.update_layout(
            title_text="Bank Statement Flow (Multi-layer: Income → Categories → Subcategories)",
            font_size=12,
            height=600
        )

        # Use stream_update counter for unique keys during streaming, otherwise use id
        chart_key = f"sankey_stream_{stream_update}" if stream_update is not None else f"sankey_{id(category_totals)}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

        # Display category summary
        st.write("### Spending by Category")
        for _, row in category_totals.iterrows():
            st.write(f"**{row['Category']}**: ${row['Amount']:,.2f}")

    # Apply date filter first
    df_filtered = filter_by_date(df, date_filter, start_date, end_date)

    if df_filtered.empty:
        st.error("No transactions found for the selected date range.")
        return

    # Show date range info
    min_date = df_filtered['Date'].min().strftime('%Y-%m-%d')
    max_date = df_filtered['Date'].max().strftime('%Y-%m-%d')
    st.info(f"Showing transactions from {min_date} to {max_date} ({len(df_filtered)} transactions)")

    # Create placeholders for live updates
    transactions_table_placeholder = st.empty()
    sankey_chart_placeholder = st.empty()

    # Process full dataset with GPT-5 only if not already done
    if st.session_state.processed_expenses is None:
        # Initial processing - apply keyword categorization
        working_df = df.copy()
        working_df['Amount'] = pd.to_numeric(working_df['Amount'], errors='coerce')
        expenses = working_df[working_df['Amount'] < 0].copy()
        expenses['Amount'] = expenses['Amount'].abs()
        expenses['Category'] = expenses['Description'].apply(categorize_transaction)

        # Add Date column to expenses
        expenses['Date'] = pd.to_datetime(
            df.loc[expenses.index, 'Transaction Date'] if 'Transaction Date' in df.columns else df.loc[expenses.index, 'Posting Date'],
            errors='coerce'
        )

        # Store in session state immediately so date filtering works
        st.session_state.processed_expenses = expenses.copy()

        # Filter expenses for current date range
        expenses_filtered = expenses[expenses['Date'].isin(df_filtered['Date'])].copy()

        # Display initial state immediately
        with transactions_table_placeholder.container():
            st.write("### All Transactions")
            st.write(f"Showing {len(expenses_filtered)} expense transactions (categories update live during GPT-5 processing)")
            table_data = expenses_filtered[['Date', 'Description', 'Amount', 'Category']].copy()
            table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m-%d')
            table_data = table_data.sort_values('Date', ascending=False)
            st.dataframe(
                table_data,
                use_container_width=True,
                height=400,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%.2f", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium")
                }
            )

        # Display initial Sankey diagram
        category_totals, sankey_data = create_sankey_from_expenses(expenses_filtered)
        with sankey_chart_placeholder.container():
            display_sankey(category_totals, sankey_data, expenses_filtered)

        # Now run GPT-5 categorization in background (only for "Other" transactions)
        other_transactions = expenses[expenses['Category'] == 'Other']['Description'].unique().tolist()
        if other_transactions and api_key:
            # Use streaming mode with automatic chunking
            def stream_update_callback(updated_expenses, stream_update=None):
                """Called during streaming to update the display."""
                # Update session state with new categorizations
                st.session_state.processed_expenses = updated_expenses.copy()

                # Filter for current date range
                updated_filtered = updated_expenses[updated_expenses['Date'].isin(df_filtered['Date'])].copy()

                # Update table
                with transactions_table_placeholder.container():
                    st.write("### All Transactions")
                    st.write(f"Showing {len(updated_filtered)} expense transactions (🔴 GPT-5 categorizing...)")
                    table_data = updated_filtered[['Date', 'Description', 'Amount', 'Category']].copy()
                    table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m-%d')
                    table_data = table_data.sort_values('Date', ascending=False)
                    st.dataframe(
                        table_data,
                        use_container_width=True,
                        height=400,
                        column_config={
                            "Date": st.column_config.TextColumn("Date", width="small"),
                            "Description": st.column_config.TextColumn("Description", width="large"),
                            "Amount": st.column_config.NumberColumn("Amount", format="$%.2f", width="small"),
                            "Category": st.column_config.TextColumn("Category", width="medium")
                        }
                    )

                # Update Sankey
                updated_totals, updated_sankey = create_sankey_from_expenses(updated_filtered)
                with sankey_chart_placeholder.container():
                    display_sankey(updated_totals, updated_sankey, updated_filtered, stream_update=stream_update)

            gpt_categories = categorize_with_gpt5_streaming(
                other_transactions,
                api_key,
                update_callback=stream_update_callback,
                expenses_df=expenses
            )
            # Update categories based on GPT-5 results
            expenses['Category'] = expenses.apply(
                lambda row: gpt_categories.get(row['Description'], row['Category'])
                if row['Category'] == 'Other' else row['Category'],
                axis=1
            )
            # Update session state with final results
            st.session_state.processed_expenses = expenses.copy()

        # Final update
        expenses = st.session_state.processed_expenses.copy()
        expenses_filtered = expenses[expenses['Date'].isin(df_filtered['Date'])].copy()

        # Final table update
        with transactions_table_placeholder.container():
            st.write("### All Transactions")
            st.write(f"Showing {len(expenses_filtered)} expense transactions ✅")
            table_data = expenses_filtered[['Date', 'Description', 'Amount', 'Category']].copy()
            table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m-%d')
            table_data = table_data.sort_values('Date', ascending=False)
            st.dataframe(
                table_data,
                use_container_width=True,
                height=400,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%.2f", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium")
                }
            )

        # Final Sankey update
        category_totals, sankey_data = create_sankey_from_expenses(expenses_filtered)
        with sankey_chart_placeholder.container():
            display_sankey(category_totals, sankey_data, expenses_filtered)

    else:
        # Use cached processed expenses
        expenses = st.session_state.processed_expenses.copy()
        expenses_filtered = expenses[expenses['Date'].isin(df_filtered['Date'])].copy()

        # Display table (no GPT-5 processing needed)
        with transactions_table_placeholder.container():
            st.write("### All Transactions")
            st.write(f"Showing {len(expenses_filtered)} expense transactions")
            table_data = expenses_filtered[['Date', 'Description', 'Amount', 'Category']].copy()
            table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m-%d')
            table_data = table_data.sort_values('Date', ascending=False)
            st.dataframe(
                table_data,
                use_container_width=True,
                height=400,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%.2f", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium")
                }
            )

        # Display Sankey
        category_totals, sankey_data = create_sankey_from_expenses(expenses_filtered)
        with sankey_chart_placeholder.container():
            display_sankey(category_totals, sankey_data, expenses_filtered)

    if category_totals.empty:
        st.error("No expenses found in the selected date range.")
        return

    # Add helpful info about multi-layer visualization
    with st.expander("ℹ️ Understanding the Multi-Layer Sankey Diagram"):
        st.markdown("""
        This diagram shows your spending in a **hierarchical structure**:

        - **Income** (left) → **Top-level Categories** (middle) → **Subcategories** (right)
        - For example: `Income → Entertainment → Concerts/Events`
        - Parent categories (like "Entertainment") show the total of all their subcategories
        - Some transactions may flow through multiple layers based on their categorization
        - Dollar amounts on nodes show the total value flowing through that category

        **Why some categories appear with no direct income flow:**
        - If you only have subcategory expenses (e.g., "Concerts/Events" under "Entertainment")
        - The flow goes: Income → Entertainment → Concerts/Events
        - This creates intermediate nodes that aggregate subcategory spending
        """)

    st.write("### Transactions by Category")

    # Get list of all categories for dropdown (built-in + subcategories + custom)
    available_categories = ALL_CATEGORIES + st.session_state.custom_categories

    for _, row in category_totals.iterrows():
        category = row['Category']
        amount = row['Amount']
        with st.expander(f"{category} (${amount:,.2f})"):
            cat_transactions = expenses_filtered[expenses_filtered['Category'] == category][['Description', 'Amount']].copy()

            st.write(f"**{len(cat_transactions)} transactions** - Click any to recategorize:")

            # Display transactions with recategorize buttons
            for idx, tx in cat_transactions.iterrows():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"{tx['Description'][:70]}")
                with col2:
                    st.write(f"${tx['Amount']:.2f}")
                with col3:
                    # Show transaction date from expenses_filtered which now has Date column
                    tx_date = expenses_filtered.loc[idx, 'Date'] if 'Date' in expenses_filtered.columns else None
                    if tx_date:
                        st.write(f"{tx_date.strftime('%Y-%m-%d')}")
                    else:
                        st.write("-")
                with col4:
                    new_category = st.selectbox(
                        "Category",
                        available_categories,
                        index=available_categories.index(category),
                        key=f"recategorize_{idx}",
                        label_visibility="collapsed"
                    )
                    if new_category != category:
                        # Save to database and session state
                        save_category_modification(tx['Description'], new_category)
                        st.session_state.category_modifications[tx['Description']] = new_category
                        st.rerun()


if __name__ == '__main__':
    main()
