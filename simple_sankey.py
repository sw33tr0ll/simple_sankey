import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI
from pydantic import BaseModel, Field
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

# Categorization prompt for LLM
CATEGORIZATION_PROMPT = """You are a financial transaction categorizer. For each transaction, provide:

1. **Category**: 1-2 SHORT words maximum (e.g., "Gas Station", "Restaurant", "Groceries", "Gym", "Theater", "Health")
2. **Parent Category**: ONE WORD, very generic (e.g., "Transport", "Food", "Entertainment", "Fitness", "Professional", "Services", "Financial")

CRITICAL RULES:
- NEVER use "Other" or "Bills" as category or parent - YOU MUST be specific and creative
- Keep categories SHORT and DIRECT: Use "Restaurant" not "Japanese Cuisine", "Groceries" not "Supermarket"
- NO hyphens or extra details: Use "Theater" not "Entertainment - Theater"
- Parent must be ONE word: "Food", "Entertainment", "Shopping", "Transport", "Health", "Fitness", "Pets", "Professional", "Services", "Financial"

HOW TO HANDLE UNCLEAR TRANSACTIONS:
- Checks → "Check Payment" / parent: "Financial"
- ATM/Withdrawals → "Cash Withdrawal" / parent: "Financial"
- Transfers → "Transfer" / parent: "Financial"
- PayPal subscriptions → Identify the service (e.g., "Streaming" for Paramount+)
- Unknown merchants → Use business type (e.g., "Local Service" / "Professional")
- Moving/Storage → "Moving" or "Storage" / parent: "Services"
- Vehicle registration → "Registration" / parent: "Transport"

Examples:
- Whole Foods → category: "Groceries", parent: "Food"
- Netflix → category: "Streaming", parent: "Entertainment"
- CHECK 235 → category: "Check Payment", parent: "Financial"
- Spirit Halloween → category: "Seasonal Store", parent: "Shopping"
- U-Haul → category: "Moving", parent: "Services"
- ATM Withdrawal → category: "Cash Withdrawal", parent: "Financial"
"""

# Database setup
DB_PATH = os.path.join(os.path.expanduser('.'), 'simple_sankey.db')

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
            category TEXT NOT NULL,
            parent_category TEXT
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

    # Try to get parent_category, but handle old databases that don't have it
    try:
        cursor.execute('SELECT description, category, parent_category FROM category_modifications')
        modifications = {row[0]: {'category': row[1], 'parent': row[2]} for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        # Old database without parent_category column
        cursor.execute('SELECT description, category FROM category_modifications')
        modifications = {row[0]: {'category': row[1], 'parent': None} for row in cursor.fetchall()}

    conn.close()
    return modifications

def save_category_modification(description, category, parent_category=None):
    """Save a transaction category modification to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO category_modifications (description, category, parent_category)
        VALUES (?, ?, ?)
    ''', (description, category, parent_category))
    conn.commit()
    conn.close()

class TransactionCategorization(BaseModel):
    model_config = {"extra": "forbid"}
    description: str
    category: str = Field(..., description="Specific category name (e.g., 'Gas', 'Coffee Shops', 'Streaming Services')")
    parent_category: str = Field(..., description="Broader parent category (e.g., 'Transport', 'Restaurants', 'Entertainment')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

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
    """Use GPT-4.1-nano with streaming output and chunked processing for reliability."""
    if not descriptions_list:
        return {}

    # Use small chunks of 3 transactions for better accuracy
    chunk_size = 3

    client = OpenAI(api_key=api_key)

    # Collect previously used categories for consistency
    existing_categories = set()
    existing_parents = set()
    if expenses_df is not None and 'Category' in expenses_df.columns:
        existing_categories = set(expenses_df['Category'].unique()) - {'Other'}
        if 'Parent_Category' in expenses_df.columns:
            existing_parents = set(expenses_df['Parent_Category'].unique()) - {'Other'}

    # Build enhanced prompt with existing categories
    categories_hint = ""
    if existing_categories:
        categories_hint = f"\n\nPREVIOUSLY USED CATEGORIES (prefer these for consistency):\n- Categories: {', '.join(sorted(existing_categories))}\n- Parents: {', '.join(sorted(existing_parents))}"

    enhanced_prompt = CATEGORIZATION_PROMPT + categories_hint

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

    # Split into chunks - keep track of descriptions in order
    chunks = [descriptions_list[i:i + chunk_size] for i in range(0, len(descriptions_list), chunk_size)]

    # Create expandable section for live categorization
    with st.expander("🔴 Live Categorization Stream", expanded=True):
        status_placeholder = st.empty()
        categorization_list = st.empty()
        progress_bar = st.progress(0)

    all_results = {}
    update_counter = 0

    for chunk_idx, chunk in enumerate(chunks):
        # Create transaction text with numbered list
        transactions_text = '\n'.join([f"{i+1}. {desc}" for i, desc in enumerate(chunk)])
        # Keep mapping of what we sent to the LLM
        chunk_descriptions = list(chunk)

        # Show chunk progress
        status_placeholder.info(f"⏳ Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} transactions)...")

        try:
            stream = client.responses.create(
                model="gpt-4.1-nano",
                instructions=enhanced_prompt,
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
                                    # Match by position in the list (more reliable than string matching)
                                    new_transactions = {}
                                    new_parents = {}
                                    for idx, t in enumerate(partial_json['transactions']):
                                        if idx >= len(chunk_descriptions):
                                            break  # LLM returned more than expected

                                        actual_desc = chunk_descriptions[idx]  # The real description from our data
                                        cat = t['category']  # Use LLM's category directly
                                        parent_cat = t.get('parent_category', cat)

                                        # Reject "Other" categorizations - force something better
                                        if cat == 'Other':
                                            st.warning(f"⚠️ AI returned 'Other' - using fallback: {actual_desc[:30]}...")
                                            # Provide a better default based on description keywords
                                            desc_lower = actual_desc.lower()
                                            if 'check' in desc_lower:
                                                cat = 'Check Payment'
                                                parent_cat = 'Financial'
                                            elif 'transfer' in desc_lower or 'ach' in desc_lower:
                                                cat = 'Transfer'
                                                parent_cat = 'Financial'
                                            elif 'atm' in desc_lower or 'withdrawal' in desc_lower:
                                                cat = 'Cash Withdrawal'
                                                parent_cat = 'Financial'
                                            elif 'paypal' in desc_lower:
                                                cat = 'Online Payment'
                                                parent_cat = 'Financial'
                                            else:
                                                cat = 'Uncategorized'
                                                parent_cat = 'Services'

                                        # Also fix "Other" parents
                                        if parent_cat == 'Other':
                                            parent_cat = cat  # Make it top-level

                                        if actual_desc not in chunk_results:
                                            # This is a new categorization
                                            new_transactions[actual_desc] = cat
                                            new_parents[actual_desc] = parent_cat
                                            chunk_results[actual_desc] = cat

                                            # Update hierarchy dynamically if we have a parent
                                            if parent_cat and parent_cat != cat and cat not in CATEGORY_HIERARCHY:
                                                CATEGORY_HIERARCHY[cat] = parent_cat

                                            # Save to database immediately
                                            save_category_modification(actual_desc, cat, parent_cat)

                                    # Show categorized transactions if we have new ones
                                    if new_transactions:
                                        all_results.update(new_transactions)
                                        cat_text = "\n".join([f"✓ {desc[:50]}... → **{cat}**"
                                                             for desc, cat in list(all_results.items())[-10:]])
                                        categorization_list.markdown(f"**Recent categorizations ({len(all_results)}/{len(descriptions_list)}):**\n\n{cat_text}")

                                        # Update expenses dataframe with new categorizations
                                        if update_callback and expenses_df is not None:
                                            for actual_desc, cat in new_transactions.items():
                                                expenses_df.loc[expenses_df['Description'] == actual_desc, 'Category'] = cat
                                                expenses_df.loc[expenses_df['Description'] == actual_desc, 'Parent_Category'] = new_parents[actual_desc]
                                            update_counter += 1
                                            update_callback(expenses_df.copy(), stream_update=update_counter)
                            except json.JSONDecodeError:
                                # Not valid JSON yet, keep accumulating
                                pass

                    elif event.type == 'response.output_text.done':
                        # Final complete text
                        if hasattr(event, 'text'):
                            full_text = event.text

            # Parse final results for this chunk - match by position
            output_json = json.loads(full_text)
            chunk_parents = {}
            for idx, t in enumerate(output_json['transactions']):
                if idx >= len(chunk_descriptions):
                    break  # LLM returned more than expected

                actual_desc = chunk_descriptions[idx]  # The real description from our data
                cat = t['category']
                parent_cat = t.get('parent_category', cat)

                # Reject "Other" categorizations - apply same fallback logic
                if cat == 'Other':
                    desc_lower = actual_desc.lower()
                    if 'check' in desc_lower:
                        cat = 'Check Payment'
                        parent_cat = 'Financial'
                    elif 'transfer' in desc_lower or 'ach' in desc_lower:
                        cat = 'Transfer'
                        parent_cat = 'Financial'
                    elif 'atm' in desc_lower or 'withdrawal' in desc_lower:
                        cat = 'Cash Withdrawal'
                        parent_cat = 'Financial'
                    elif 'paypal' in desc_lower:
                        cat = 'Online Payment'
                        parent_cat = 'Financial'
                    else:
                        cat = 'Uncategorized'
                        parent_cat = 'Services'

                # Fix "Other" parents
                if parent_cat == 'Other':
                    parent_cat = cat

                chunk_results[actual_desc] = cat
                chunk_parents[actual_desc] = parent_cat

                # Update hierarchy dynamically
                if parent_cat and parent_cat != cat and cat not in CATEGORY_HIERARCHY:
                    CATEGORY_HIERARCHY[cat] = parent_cat

            all_results.update(chunk_results)

            # Update expenses_df with final chunk results
            if update_callback and expenses_df is not None:
                for desc, cat in chunk_results.items():
                    expenses_df.loc[expenses_df['Description'] == desc, 'Category'] = cat
                    expenses_df.loc[expenses_df['Description'] == desc, 'Parent_Category'] = chunk_parents[desc]
                    # Update running sets for next chunks
                    existing_categories.add(cat)
                    existing_parents.add(chunk_parents[desc])

                # Rebuild hint for next chunk
                if existing_categories:
                    categories_hint = f"\n\nPREVIOUSLY USED CATEGORIES (prefer these for consistency):\n- Categories: {', '.join(sorted(existing_categories))}\n- Parents: {', '.join(sorted(existing_parents))}"
                    enhanced_prompt = CATEGORIZATION_PROMPT + categories_hint

                update_counter += 1
                update_callback(expenses_df.copy(), stream_update=update_counter)

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
    """Use GPT-4.1-nano with chunked requests (multiple smaller batches)."""
    if not descriptions_list:
        return {}

    # Use small chunks of 3 transactions for better accuracy
    chunk_size = 3

    client = OpenAI(api_key=api_key)

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
        chunk_descriptions = list(chunk)  # Keep original descriptions

        try:
            response = client.responses.create(
                model="gpt-4.1-nano",
                instructions=CATEGORIZATION_PROMPT,
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
            for t_idx, t in enumerate(output_json['transactions']):
                if t_idx >= len(chunk_descriptions):
                    break

                actual_desc = chunk_descriptions[t_idx]  # Match by position
                cat = t['category']
                parent_cat = t.get('parent_category', cat)

                results[actual_desc] = cat

                # Update hierarchy dynamically
                if parent_cat and parent_cat != cat and cat not in CATEGORY_HIERARCHY:
                    CATEGORY_HIERARCHY[cat] = parent_cat

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

    # Use GPT-4.1-nano to categorize "Other" transactions
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

    # OpenAI API Key
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

    # Debug mode
    st.session_state.show_debug = st.sidebar.checkbox("🔍 Show Debug Info", value=False)

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

        # Build a complete hierarchy map from both CATEGORY_HIERARCHY and Parent_Category column
        hierarchy_map = dict(CATEGORY_HIERARCHY)  # Start with hardcoded hierarchy

        # Add AI-generated parent relationships from the dataframe
        if 'Parent_Category' in expenses_df.columns:
            for _, row in expenses_df.iterrows():
                cat = row['Category']
                parent = row.get('Parent_Category')
                # Skip invalid parents: Other, same as category, or empty
                if parent and parent != cat and parent not in ['Other', ''] and cat not in hierarchy_map:
                    hierarchy_map[cat] = parent

        # Debug: show unique category→parent mappings
        if st.session_state.get('show_debug'):
            with st.expander("🔍 Category Hierarchy Debug"):
                unique_mappings = expenses_df[['Category', 'Parent_Category']].drop_duplicates().sort_values('Category')
                st.write("**Category → Parent mappings in data:**")
                st.dataframe(unique_mappings)

        def get_ancestry_chain(category):
            """Get the full chain of parents for a category (from root to category)."""
            chain = [category]
            current = category
            seen = set([category])  # Prevent infinite loops
            while current in hierarchy_map:
                parent = hierarchy_map[current]
                if parent in seen:  # Circular reference, stop
                    break
                chain.insert(0, parent)
                seen.add(parent)
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
            for desc, mod_data in st.session_state.category_modifications.items():
                # Handle both old format (string) and new format (dict)
                if isinstance(mod_data, dict):
                    new_cat = mod_data['category']
                    parent = mod_data.get('parent') or CATEGORY_HIERARCHY.get(new_cat, new_cat)
                else:
                    # Old format: just a category string
                    new_cat = mod_data
                    parent = CATEGORY_HIERARCHY.get(new_cat, new_cat)

                expenses.loc[expenses['Description'] == desc, 'Category'] = new_cat
                if 'Parent_Category' in expenses.columns:
                    expenses.loc[expenses['Description'] == desc, 'Parent_Category'] = parent

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
        st.plotly_chart(fig, width='stretch', key=chart_key)

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

    # Process full dataset with GPT-4.1-nano only if not already done
    if st.session_state.processed_expenses is None:
        # Initial processing - apply keyword categorization
        working_df = df.copy()
        working_df['Amount'] = pd.to_numeric(working_df['Amount'], errors='coerce')
        expenses = working_df[working_df['Amount'] < 0].copy()
        expenses['Amount'] = expenses['Amount'].abs()
        expenses['Category'] = expenses['Description'].apply(categorize_transaction)
        expenses['Parent_Category'] = 'Other'  # Will be updated by AI

        # Apply existing categorizations from database to skip AI re-processing
        for desc, mod_data in st.session_state.category_modifications.items():
            if isinstance(mod_data, dict):
                cat = mod_data['category']
                parent = mod_data.get('parent') or CATEGORY_HIERARCHY.get(cat, cat)
            else:
                cat = mod_data
                parent = CATEGORY_HIERARCHY.get(cat, cat)

            # Update if this description exists in current expenses
            mask = expenses['Description'] == desc
            if mask.any():
                expenses.loc[mask, 'Category'] = cat
                expenses.loc[mask, 'Parent_Category'] = parent

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
            st.write(f"Showing {len(expenses_filtered)} expense transactions (categories update live during GPT-4.1-nano processing)")
            table_data = expenses_filtered[['Date', 'Description', 'Amount', 'Category', 'Parent_Category']].copy()
            table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m-%d')
            table_data = table_data.sort_values('Date', ascending=False)
            st.dataframe(
                table_data,
                width='stretch',
                height=400,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%.2f", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Parent_Category": st.column_config.TextColumn("Parent", width="small")
                }
            )

        # Display initial Sankey diagram
        category_totals, sankey_data = create_sankey_from_expenses(expenses_filtered)
        with sankey_chart_placeholder.container():
            display_sankey(category_totals, sankey_data, expenses_filtered)

        # Now run GPT-4.1-nano categorization on uncategorized transactions only
        # Get transactions that aren't already in the database
        uncategorized_mask = ~expenses['Description'].isin(st.session_state.category_modifications.keys())
        uncategorized_transactions = expenses.loc[uncategorized_mask, 'Description'].unique().tolist()

        if uncategorized_transactions and api_key:
            st.info(f"🤖 Running AI categorization on {len(uncategorized_transactions)} new transactions ({len(st.session_state.category_modifications)} already in database)")

            # Use streaming mode with automatic chunking
            def stream_update_callback(updated_expenses, stream_update=None):
                """Called during streaming to update the display."""
                try:
                    # Update session state with new categorizations
                    st.session_state.processed_expenses = updated_expenses.copy()

                    # Filter for current date range
                    updated_filtered = updated_expenses[updated_expenses['Date'].isin(df_filtered['Date'])].copy()

                    # Update table
                    with transactions_table_placeholder.container():
                        st.write("### All Transactions")
                        st.write(f"Showing {len(updated_filtered)} expense transactions (🔴 GPT-4.1-nano categorizing...)")
                        table_data = updated_filtered[['Date', 'Description', 'Amount', 'Category', 'Parent_Category']].copy()
                        table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m-%d')
                        table_data = table_data.sort_values('Date', ascending=False)
                        st.dataframe(
                            table_data,
                            width='stretch',
                            height=400,
                            column_config={
                                "Date": st.column_config.TextColumn("Date", width="small"),
                                "Description": st.column_config.TextColumn("Description", width="large"),
                                "Amount": st.column_config.NumberColumn("Amount", format="$%.2f", width="small"),
                                "Category": st.column_config.TextColumn("Category", width="medium"),
                                "Parent_Category": st.column_config.TextColumn("Parent", width="small")
                            }
                        )

                    # Update Sankey
                    updated_totals, updated_sankey = create_sankey_from_expenses(updated_filtered)
                    with sankey_chart_placeholder.container():
                        display_sankey(updated_totals, updated_sankey, updated_filtered, stream_update=stream_update)
                except Exception as e:
                    st.error(f"Error in stream_update_callback: {str(e)}")

            # Streaming function updates expenses in place and session state via callbacks
            categorize_with_gpt5_streaming(
                uncategorized_transactions,
                api_key,
                update_callback=stream_update_callback,
                expenses_df=expenses
            )
            # Session state already updated via callbacks during streaming
            # Just ensure it's saved one final time
            st.session_state.processed_expenses = expenses.copy()
        else:
            # All transactions already categorized from database
            st.success(f"✅ All {len(expenses['Description'].unique())} unique transactions already categorized from database!")

        # Final update
        expenses = st.session_state.processed_expenses.copy()

        # Ensure Parent_Category column exists (for backward compatibility)
        if 'Parent_Category' not in expenses.columns:
            expenses['Parent_Category'] = 'Other'

        expenses_filtered = expenses[expenses['Date'].isin(df_filtered['Date'])].copy()

        # Final table update
        with transactions_table_placeholder.container():
            st.write("### All Transactions")
            st.write(f"Showing {len(expenses_filtered)} expense transactions ✅")
            table_data = expenses_filtered[['Date', 'Description', 'Amount', 'Category', 'Parent_Category']].copy()
            table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m-%d')
            table_data = table_data.sort_values('Date', ascending=False)
            st.dataframe(
                table_data,
                width='stretch',
                height=400,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%.2f", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Parent_Category": st.column_config.TextColumn("Parent", width="small")
                }
            )

        # Final Sankey update
        category_totals, sankey_data = create_sankey_from_expenses(expenses_filtered)
        with sankey_chart_placeholder.container():
            display_sankey(category_totals, sankey_data, expenses_filtered)

    else:
        # Use cached processed expenses
        expenses = st.session_state.processed_expenses.copy()

        # Ensure Parent_Category column exists (for backward compatibility)
        if 'Parent_Category' not in expenses.columns:
            expenses['Parent_Category'] = 'Other'

        expenses_filtered = expenses[expenses['Date'].isin(df_filtered['Date'])].copy()

        # Display table (no GPT-5 processing needed)
        with transactions_table_placeholder.container():
            st.write("### All Transactions")
            st.write(f"Showing {len(expenses_filtered)} expense transactions")
            table_data = expenses_filtered[['Date', 'Description', 'Amount', 'Category', 'Parent_Category']].copy()
            table_data['Date'] = table_data['Date'].dt.strftime('%Y-%m-%d')
            table_data = table_data.sort_values('Date', ascending=False)
            st.dataframe(
                table_data,
                width='stretch',
                height=400,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Amount": st.column_config.NumberColumn("Amount", format="$%.2f", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Parent_Category": st.column_config.TextColumn("Parent", width="small")
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

    # Get list of all categories for dropdown (built-in + subcategories + custom + LLM-generated)
    available_categories = sorted(set(
        list(ALL_CATEGORIES) +
        st.session_state.custom_categories +
        expenses_filtered['Category'].unique().tolist()
    ))

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
                    # Safely get index, defaulting to 0 if category not found
                    try:
                        default_index = available_categories.index(category)
                    except ValueError:
                        default_index = 0

                    new_category = st.selectbox(
                        "Category",
                        available_categories,
                        index=default_index,
                        key=f"recategorize_{idx}",
                        label_visibility="collapsed"
                    )
                    if new_category != category:
                        # Get parent from hierarchy
                        parent = CATEGORY_HIERARCHY.get(new_category, new_category)
                        # Save to database and session state
                        save_category_modification(tx['Description'], new_category, parent)
                        st.session_state.category_modifications[tx['Description']] = {'category': new_category, 'parent': parent}
                        # Clear processed expenses to force reload with new categories
                        st.session_state.processed_expenses = None
                        st.rerun()


if __name__ == '__main__':
    main()
