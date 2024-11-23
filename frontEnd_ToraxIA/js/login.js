document.getElementById('loginForm').addEventListener('submit', async function (e) {
    e.preventDefault();


    // Limpiar localStorage antes de guardar los datos del nuevo usuario
    localStorage.clear();


    // Obtener los valores de los campos
    const username = document.getElementById('username').value;
    const passwordInput = document.getElementById('passwordInput').value;

    let nombreMedico = '';
    let matriculaMedico = '';
    let nombreTecnico = '';
    let nombreAdmin = '';

    switch (username) {
        case "Rodriguez":
            nombreMedico = "Dr. Rodriguez";
            matriculaMedico = "MP1234";
            // Guardar los datos en localStorage o sessionStorage
            localStorage.setItem('nombreMedico', nombreMedico);
            localStorage.setItem('matriculaMedico', matriculaMedico);
            window.location.href = './searchMedico.html';
            break;

        case "Garcia":
            nombreMedico = "Dra. Garcia";
            matriculaMedico = "MP5678";
            // Guardar los datos en localStorage o sessionStorage
            localStorage.setItem('nombreMedico', nombreMedico);
            localStorage.setItem('matriculaMedico', matriculaMedico);
            window.location.href = './searchMedico.html';
            break;

        case "Perez":
            nombreMedico = "Dr. Perez";
            matriculaMedico = "MP9101";
            // Guardar los datos en localStorage o sessionStorage
            localStorage.setItem('nombreMedico', nombreMedico);
            localStorage.setItem('matriculaMedico', matriculaMedico);
            window.location.href = './searchMedico.html';
            break;

        case "Tecnico":
            nombreTecnico = "Tec. Roberto Hernandez";
            // Guardar los datos en localStorage o sessionStorage
            localStorage.setItem('nombreTecnico', nombreTecnico);
            window.location.href = './predictions.html';
            break;
        case "Admin":
            nombreAdmin = "Superusuario";
            // Guardar los datos en localStorage o sessionStorage
            localStorage.setItem('nombreAdmin', nombreAdmin);
            window.location.href = './vistaAdministrador.html';
            break;
        default:
            alert("Usuario y/o contraseña incorrectos");
    }
    document.getElementById('username').value = '';

});

