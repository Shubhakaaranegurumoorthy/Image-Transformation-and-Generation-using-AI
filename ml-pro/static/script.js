// Toggle between Login and Signup forms
function toggleForm() {
    document.getElementById("login-form").classList.toggle("hidden");
    document.getElementById("signup-form").classList.toggle("hidden");
}
    
    // Signup function
function signup() {
    const email = document.getElementById("signup-email").value.trim();
    const password = document.getElementById("signup-password").value.trim();
    if (!validateEmail(email)) {
        alert("Please enter a valid email.");
        return;
    }
    if (password.length < 6) {
        alert("Password must be at least 6 characters long.");
        return;
    }
    
    localStorage.setItem("user", JSON.stringify({ email, password }));
    alert("Signup successful! Please login.");
    toggleForm();
}
    
// Login function
function login() {
    const email = document.getElementById("login-email").value.trim();
    const password = document.getElementById("login-password").value.trim();
    const user = JSON.parse(localStorage.getItem("user"));
    
    if (!user) {
        alert("No account found. Please sign up first.");
    return;
    }
    
    if (user.email === email && user.password === password) {
        // alert("Login successful!");
        window.location.href = "pages/ml-features.html"; // Redirect to ML Features page
    } else {
        alert("Invalid credentials. Please try again.");
    }
}
    
    // Email validation function
function validateEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}
    

