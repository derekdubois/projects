import os

from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import apology, login_required, lookup, usd, check_password

# Configure application
app = Flask(__name__)

# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response




@app.route("/")
@login_required
def index():
    """Show portfolio of stocks"""
    stocks = db.execute("SELECT symbol, SUM(shares) as total_shares FROM transactions WHERE user_id = :user_id GROUP BY symbol HAVING total_shares > 0",
                        user_id=session["user_id"])

    #retrieve user cash balance
    cash = db.execute("SELECT cash FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["cash"]

    # set variables for totals
    sum_total = cash
    end_total = cash

    # add in prices and values
    for stock in stocks:
        quote = lookup(stock["symbol"])
        stock["name"] = quote["name"]
        stock["price"] = quote["price"]
        stock["value"] = stock["price"] * stock["total_shares"]
        sum_total += stock["value"]
        end_total += stock["value"]

    return render_template("index.html", stocks=stocks, cash=cash, sum_total=sum_total, end_total=end_total)





@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    """Buy shares of stock"""
    if request.method == "POST":
        symbol = request.form.get("symbol").upper()
        shares = request.form.get("shares")
        if not symbol:
            return apology("Please provide stock symbol")
        elif not shares or not shares.isdigit() or int(shares) <= 0:
            return apology("Please provide a positive number of shares")

        #check for symbol
        quote = lookup(symbol)
        if quote is None:
            return apology("No Symbol Found")

        #calculate total buy
        price = quote["price"]
        total_cost = int(shares) * price
        cash = db.execute("SELECT cash FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["cash"]

        #see if enough cash
        if cash < total_cost:
            return apology("Not enough funds")

        #update user cash in database
        db.execute("UPDATE users SET cash = cash - :total_cost WHERE id = :user_id",
                   total_cost=total_cost, user_id=session["user_id"])

        #add buy to buy history
        db.execute("INSERT INTO transactions (user_id, symbol, shares, price) VALUES (:user_id, :symbol, :shares, :price)",
                   user_id=session["user_id"], symbol=symbol, shares=shares, price=price)

        #show alert
        flash(f"Purchased {shares} shares of {symbol} for {usd(total_cost)}")
        return redirect("/")

    else:
        return render_template("buy.html")




@app.route("/history")
@login_required
def history():
    """Show history of transactions"""
    #check database for user transactions
    user_transactions = db.execute(
        "SELECT * FROM transactions WHERE user_id = :user_id ORDER BY timestamp DESC", user_id=session["user_id"])

    #redirect to history page
    return render_template("history.html", transactions=user_transactions)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Ensure username was submitted
        if not username:
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not password:
            return apology("must provide password", 403)

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = ?", username)

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], password):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote."""
    if request.method == "POST":
        symbol = request.form.get("symbol")
        quote = lookup(symbol)
        if not quote:
            return apology("Input valid symbol", 400)
        return render_template("quote.html", quote=quote)
    else:
        return render_template("quote.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    # clear user_id
    session.clear()

    # post request
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        #check for username
        if not username:
            return apology("Please provide username", 400)

        #check for password
        elif not password:
            return apology("Please provide password", 400)

        #check for valid password
        elif not check_password(password):
            return apology("Password must be at least for characters and contain 1 upper, 1 lower & 1 special character", 400)

        #check for password confirmation
        elif not confirmation:
            return apology("Please confirm password", 400)

        #check for matching passwords
        elif password != confirmation:
            return apology("Please provide matching passwords", 400)

        #check database for user
        rows = db.execute("SELECT * FROM users WHERE username = ?", username)

        #check to make sure user doesn't already exist
        if len(rows) != 0:
            return apology("User already exists", 400)

        user_pass = generate_password_hash(password)

        #insert user into database
        db.execute("INSERT INTO users (username, hash) VALUES(?, ?)",
                    username, user_pass)

        #check database for recently inserted user
        rows = db.execute("SELECT * FROM users WHERE username = ?", username)

        #Store logged in user
        session["user_id"] = rows[0]["id"]

        #redirect user back to homepage
        return redirect("/")


    #send user back to homepage
    else:
        return render_template("register.html")




@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock"""
    #display user stocks
    stocks = db.execute("SELECT symbol, SUM(shares) as total_shares FROM transactions WHERE user_id = :user_id GROUP BY symbol HAVING total_shares > 0",
                        user_id=session["user_id"])

    #for user submit
    if request.method == "POST":
        symbol = request.form.get("symbol").upper()
        shares = request.form.get("shares")
        if not symbol:
            return apology("Please provide symbol")
        elif not shares or not shares.isdigit() or int(shares) <= 0:
            return apology("Please specify a positive # of shares")
        else:
            shares = int(shares)

        for stock in stocks:
            if stock["symbol"] == symbol:
                if stock["total_shares"] < shares:
                    return apology("Insufficient shares")
                else:
                    # get stock quote
                    quote = lookup(symbol)
                    if quote is None:
                        return apology("Input valid symbol")
                    price = quote["price"]
                    end_sale = shares * price

                    #update SQL
                    db.execute("UPDATE users SET cash = cash + :end_sale WHERE id = :user_id",
                               end_sale=end_sale, user_id=session["user_id"])

                    #update history
                    db.execute("INSERT INTO transactions (user_id, symbol, shares, price) VALUES (:user_id, :symbol, :shares, :price)",
                               user_id=session["user_id"], symbol=symbol, shares=-shares, price=price)

                    flash(f"You sold {shares} shares of {symbol} for {usd(end_sale)}.")
                    return redirect("/")

        return apology("No symbol found")
    #generic page visit
    else:
        return render_template("sell.html", stocks=stocks)


