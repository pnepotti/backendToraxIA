const baseUrl = 'http://localhost:8000'; // Cambiar en producción
// Obtener el modal y elementos
const modal = document.getElementById('myModal');
const modalImg = document.getElementById('modal-image');
const closeBtn = document.getElementsByClassName('close')[0];
const patientNameContainer = document.getElementById('patient-name-container');
const container = document.getElementById('radiography-container');
const diagnosticar = document.getElementById('diagnosticar');
const otraEnfermedad = document.getElementById('otraEnfermedad');
const selectElement = document.getElementById("enfermedadVal");
let diagnostico = "";


async function actualizarRadiografia(idRx) {
    try {
        const response = await fetch(`${baseUrl}/diagnostics/api/images-by-idRx/?idRx=${idRx}`);

        const data = await response.json();

        if (response.ok) {
            document.getElementById(`validarEnfermedad-${idRx}`).innerHTML = `
                <strong>Diagnóstico confirmado: </strong>${data.radiography.diagnostico}<br><br>
            `;
        } else {
            console.error('Error al obtener el diagnóstico:', data.error);
        }
    } catch (error) {
        console.error('Error en la solicitud:', error);
    }
}



document.addEventListener('DOMContentLoaded', function () {
    const notificationBadge = document.getElementById('notificationBadge');
    const notificationIcon = document.getElementById('notificationIcon'); // Actualizado a 'notificationIcon'
    const matricula = localStorage.getItem('matriculaMedico'); // Recuperar la matrícula correctamente


    async function checkNotifications() {
        try {
            const response = await fetch(`${baseUrl}/diagnostics/api/images-by-matricula-null-diagnostic/?matricula=${matricula}`);

            if (!response.ok) {
                if (response.status === 404) {
                    console.log('No se encontraron radiografías.');
                } else {
                    throw new Error(`Error en la solicitud: ${response.status}`);
                }
                return;
            }

            const data = await response.json(); // Asignar los datos a la variable data aquí

            // Verificar si los datos no están vacíos
            if (data.radiographies && data.radiographies.length > 0) {
                notificationBadge.classList.add('active'); // Mostrar el puntito rojo
            } else {
                notificationBadge.classList.remove('active'); // Ocultar el puntito rojo si no hay notificaciones
            }

            // Manejar el clic en el ícono de notificaciones una vez que los datos están disponibles
            notificationIcon.addEventListener('click', function () {
                // Ocultar el punto rojo al hacer clic en la campana
                notificationBadge.classList.remove('active');
                notificationIcon.style.pointerEvents = "none";
                // Verificar si los datos están disponibles
                if (!data || !data.radiographies || data.radiographies.length === 0) {
                    alert('No hay radiografías disponibles para mostrar.');
                    return;
                }

                const container = document.getElementById('radiography-container');
                container.innerHTML = ''; // Limpiar el contenedor

                // Validaciones básicas
                if (!matricula) {
                    alert('Por favor, ingrese como médico.');
                    return;
                }

                patientNameContainer.innerHTML = `<h3>Radiografías sin diagnosticar</h3>`;

                // Mostrar las radiografías usando la función displayRadiographies
                displayRadiographies(data.radiographies);

                // Mostrar todas las imágenes al inicio
                const elementos = document.getElementsByClassName("card-image");
                for (let i = 0; i < elementos.length; i++) {
                    elementos[i].style.display = "block";
                }
            });

        } catch (error) {
            console.error('Error al verificar notificaciones:', error);
        }
    }

    // Llamar a la función para verificar las notificaciones al cargar la página
    checkNotifications();
});


document.getElementById('search-form-medico').addEventListener('submit', async function (e) {
    e.preventDefault();

    patientNameContainer.innerHTML = '';
    container.innerHTML = ''; // Limpiar el contenedor


    const matricula = localStorage.getItem('matriculaMedico');

    const selectedValue = document.getElementById('opcionDeBusqueda').value;
    const searchValue = document.getElementById('filtroDeBusqueda').value;

    // Validaciones básicas
    if (!searchValue) {
        alert('Por favor, completa el campo');
        return;
    }


    if (selectedValue === 'dni') {
        console.log('Buscar por DNI');
        try {

            const response = await fetch(`${baseUrl}/diagnostics/api/images/?dni=${searchValue}`);

            if (!response.ok) {
                if (response.status === 404) {
                    container.innerHTML = '<p>No se encontraron radiografías.</p>';
                } else {
                    throw new Error(`Error en la solicitud: ${response.status}`);
                }
                return;
            }

            const data = await response.json();
            // Mostrar el nombre del paciente, si está en la respuesta

            const patientName = data.radiographies[0].patient_name || "Paciente no encontrado"; // Cambiar esto si el nombre tiene otro campo
            patientNameContainer.innerHTML = `<h3>Paciente: ${patientName}</h3>`;

            displayRadiographiesDNI(data.radiographies);

        } catch (error) {
            console.error('Error en la búsqueda:', error);
        } finally {
            // Mostrar todas las imágenes al inicio
            const elementos = document.getElementsByClassName("card-image");
            for (let i = 0; i < elementos.length; i++) {
                elementos[i].style.display = "block";
            }
        }



    } else if (selectedValue === 'idRx') {
        console.log('Buscar por Matrícula del médico');
        try {
            const response = await fetch(`${baseUrl}/diagnostics/api/images-by-idRx/?idRx=${searchValue}`);
            if (!response.ok) {
                if (response.status === 404 || response.status === 400) {
                    container.innerHTML = '<p>No se encontró la radiografía.</p>';
                } else {
                    throw new Error(`Error en la solicitud: ${response.status}`);
                }
                return;
            }


            const data = await response.json();



            displayRadiographyIdRx(data.radiography);

        } catch (error) {
            console.error('Error en la búsqueda:', error);
        } finally {
            // Mostrar todas las imágenes al inicio
            const elementos = document.getElementsByClassName("card-image");
            for (let i = 0; i < elementos.length; i++) {
                elementos[i].style.display = "block";
            }
        }

    }
});

function displayRadiographiesDNI(radiographies) {
    const container = document.getElementById('radiography-container');
    container.innerHTML = ''; // Limpiar el contenedor

    if (radiographies.length === 0) {
        container.innerHTML = '<p>No se encontraron radiografías para este paciente.</p>';
        return;
    }

    radiographies.forEach(radiography => {
        const radiographyElement = document.createElement('div');
        radiographyElement.classList.add('card-image');

        const fullImageUrl = `${baseUrl}/media/${radiography.image_url.split('media/')[1]}`;
        console.log(radiography.diagnostico);
        const predictionsList = radiography.predictions.map(prediction => `
            <li>
                <br>
                <strong>Enfermedad:</strong> ${prediction.disease}<br>
                <strong>Probabilidad:</strong> ${(prediction.probability * 100).toFixed(1)}%<br>
                <strong>Confianza:</strong> ${prediction.confidence > 0.85 ? '- Alta' : '- Baja'}<br>
                <strong>Incertidumbre:</strong> ${prediction.entropy > 0.4 ? '- Alta' : '- Baja'}<br><br>
                <strong>Comentario:</strong>${radiography.descripcion || 'No disponible'}<br><br>
                
                ${radiography.diagnostico === null ? `<br><br>
                <div id="validarEnfermedad-${radiography.radiography_id}">
                <strong>Validar diagnóstico: </strong>
                <select id="enfermedadVal-${radiography.radiography_id}" class="btn">
        <option value="" disabled selected>Seleccionar enfermedad</option>
        <option value="Normal">Normal</option>
        <option value="Neumonia">Neumonia</option>
        <option value="Covid19">Covid</option>
        <option value="Tuberculosis">Tuberculosis</option>
        <option value="Neumothorax">Neumothorax</option>
        <option value="Otro">Otro</option>
                </select>
            <div id="otraEnfermedad-${radiography.radiography_id}" style="display: none;">
            <strong>Enfermedad: </strong><br>
            <input type="text" class="enfermedadValidada" id="enfermedadValidada-${radiography.radiography_id}" placeholder="Ingresar enfermedad">
            </div>
            <button class="btn diagnosticar-btn" style="width:50%" id="diagnosticar-${radiography.radiography_id}" data-radiography-id="${radiography.radiography_id}">Diagnosticar</button>
            </div>

                ` : `<strong>Diagnóstico confirmado: </strong>${radiography.diagnostico}<br><br>`}
            </li>
        `).join('');

        radiographyElement.innerHTML = `
            <img src="${fullImageUrl}" alt="Radiografía ID: ${radiography.radiography_id}" class="thumbnail" onclick="openModal('${fullImageUrl}')">
            <ul>
                <li><strong>ID de radiografia:</strong> ${radiography.radiography_id}</li>
                <li><strong>Médico:</strong> ${radiography.doctor_name}</li>
                <li><strong>Fecha:</strong> ${new Date(radiography.uploaded_at).toLocaleDateString()}</li>
                ${predictionsList}
            </ul>
        `;

        container.appendChild(radiographyElement);
    });
}

// Mostrar las radiografías en el contenedor
function displayRadiographies(radiographies) {
    const container = document.getElementById('radiography-container');
    container.innerHTML = ''; // Limpiar el contenedor

    if (radiographies.length === 0) {
        container.innerHTML = '<p>No se encontraron radiografías para este paciente</p>';
        return;
    }

    radiographies.forEach(radiography => {
        const radiographyElement = document.createElement('div');
        radiographyElement.classList.add('card-image');

        const fullImageUrl = `${baseUrl}/media/${radiography.image_url.split('media/')[1]}`;
        console.log(radiography.diagnostico);
        const predictionsList = radiography.predictions.map(prediction => `
            <li>
                <br>
                <strong>Enfermedad:</strong> ${prediction.disease}<br>
                <strong>Probabilidad:</strong> ${(prediction.probability * 100).toFixed(1)}%<br>
                <strong>Confianza:</strong> ${prediction.confidence > 0.85 ? '- Alta' : '- Baja'}<br>
                <strong>Incertidumbre:</strong> ${prediction.entropy > 0.4 ? '- Alta' : '- Baja'}<br><br>
                <strong>Comentario:</strong>${radiography.descripcion || 'No disponible'}<br><br>
                ${radiography.diagnostico === null ? `<br><br>
                <div id="validarEnfermedad-${radiography.radiography_id}">
                <strong>Validar diagnóstico: </strong>
                <select id="enfermedadVal-${radiography.radiography_id}" class="btn">
        <option value="" disabled selected>Seleccionar enfermedad</option>
        <option value="Normal">Normal</option>
        <option value="Neumonia">Neumonia</option>
        <option value="Covid19">Covid</option>
        <option value="Tuberculosis">Tuberculosis</option>
        <option value="Neumothorax">Neumothorax</option>
        <option value="Otro">Otro</option>
                </select>
            <div id="otraEnfermedad-${radiography.radiography_id}" style="display: none;">
            <strong>Enfermedad: </strong><br>
            <input type="text" class="enfermedadValidada" id="enfermedadValidada-${radiography.radiography_id}" placeholder="Ingresar enfermedad">
            </div>
            <button class="btn diagnosticar-btn" style="width:50%" id="diagnosticar-${radiography.radiography_id}" data-radiography-id="${radiography.radiography_id}">Diagnosticar</button>
            </div>
                ` : `<strong>Diagnóstico confirmado: </strong>${radiography.diagnostico}<br><br>`}
            </li>
        `).join('');

        radiographyElement.innerHTML = `
            <img src="${fullImageUrl}" alt="Radiografía ID: ${radiography.radiography_id}" class="thumbnail" onclick="openModal('${fullImageUrl}')">
            <ul>
                <li><strong>Paciente:</strong> ${radiography.patient_name}</li>
                <li><strong>ID de radiografia:</strong> ${radiography.radiography_id}</li>
                <li><strong>Fecha:</strong> ${new Date(radiography.uploaded_at).toLocaleDateString()}</li>
                ${predictionsList}
            </ul>
        `;

        container.appendChild(radiographyElement);
    });
}


function displayRadiographyIdRx(radiography) {
    const container = document.getElementById('radiography-container');
    container.innerHTML = ''; // Limpiar el contenedor

    // Si no se encuentra la radiografía (el objeto es nulo o indefinido)
    if (!radiography) {
        container.innerHTML = '<p>No se encontró la radiografía.</p>';
        return;
    }

    const radiographyElement = document.createElement('div');
    radiographyElement.classList.add('card-image');

    // Usar la URL completa de la imagen
    const fullImageUrl = radiography.image_url;

    // Verificar si hay predicciones disponibles
    const predictionsList = radiography.predictions && radiography.predictions.length > 0
        ? radiography.predictions.map(prediction => `
            <li>
                <strong>Enfermedad:</strong> ${prediction.disease}<br>
                <strong>Probabilidad:</strong> ${(prediction.probability * 100).toFixed(1)}%<br>
                <strong>Confianza:</strong> ${prediction.confidence > 0.85 ? '- Alta' : '- Baja'}<br>
                <strong>Incertidumbre:</strong> ${prediction.entropy > 0.4 ? '- Alta' : '- Baja'}<br><br>
                <strong>Comentario:</strong> ${radiography.descripcion || 'No disponible'}<br><br>
                ${radiography.diagnostico === null ? `<br><br>
                <div id="validarEnfermedad-${radiography.radiography_id}">
                <strong>Validar diagnóstico: </strong>
                <select id="enfermedadVal-${radiography.radiography_id}" class="btn">
        <option value="" disabled selected>Seleccionar enfermedad</option>
        <option value="Normal">Normal</option>
        <option value="Neumonia">Neumonia</option>
        <option value="Covid19">Covid</option>
        <option value="Tuberculosis">Tuberculosis</option>
        <option value="Neumothorax">Neumothorax</option>
        <option value="Otro">Otro</option>
                </select>
            <div id="otraEnfermedad-${radiography.radiography_id}" style="display: none;">
            <strong>Enfermedad: </strong><br>
            <input type="text" class="enfermedadValidada" id="enfermedadValidada-${radiography.radiography_id}" placeholder="Ingresar enfermedad">
            </div>
            <button class="btn diagnosticar-btn" style="width:50%" id="diagnosticar-${radiography.radiography_id}" data-radiography-id="${radiography.radiography_id}">Diagnosticar</button>
            </div>
                ` : `<strong>Diagnóstico confirmado: </strong>${radiography.diagnostico}<br><br>`}
            </li>
        `).join('')
        : '<li>No hay predicciones disponibles.</li>';

    // Insertar el contenido HTML en la tarjeta de la radiografía
    radiographyElement.innerHTML = `
        <img src="${fullImageUrl}" alt="Radiografía ID: ${radiography.radiography_id}" class="thumbnail" onclick="openModal('${fullImageUrl}')">
        <ul>
            <li><strong>ID de radiografía:</strong> ${radiography.radiography_id}</li>
            <li><strong>Médico:</strong> ${radiography.doctor_name || 'No disponible'}</li>
            <li><strong>Paciente:</strong> ${radiography.patient_name || 'No disponible'}</li>
            <li><strong>Diagnóstico:</strong> ${radiography.diagnostico || 'No disponible'}</li>
            <li><strong>Fecha:</strong> ${new Date(radiography.uploaded_at).toLocaleDateString()}</li><br>
            ${predictionsList}
        </ul>
    `;

    // Añadir la tarjeta al contenedor
    container.appendChild(radiographyElement);
}

// Abrir modal para mostrar la imagen
function openModal(imageSrc) {
    modal.style.display = "flex";
    modalImg.src = imageSrc;
    document.body.classList.add('modal-open'); // Deshabilitar el scroll del body
}

// Cerrar modal
closeBtn.onclick = function () {
    modal.style.display = "none";
    document.body.classList.remove('modal-open'); // Habilitar el scroll del body
}

// Zoom y desplazamiento de la imagen en el modal
let isZoomed = false;

modalImg.onclick = function (e) {
    const rect = modalImg.getBoundingClientRect(); // Obtener las coordenadas de la imagen visible
    const offsetX = e.clientX - rect.left; // Coordenada X dentro de la imagen visible
    const offsetY = e.clientY - rect.top;  // Coordenada Y dentro de la imagen visible

    // Calcular el porcentaje del clic dentro de la imagen
    const xPercent = (offsetX / rect.width) * 100;
    const yPercent = (offsetY / rect.height) * 100;

    if (isZoomed) {
        // Restablecer el zoom
        modalImg.style.transform = "scale(1)";
        modalImg.style.transformOrigin = "center center"; // Volver a centrar la imagen
        modalImg.style.cursor = "zoom-in";
        modalImg.style.width = "800px";
        isZoomed = false;
    } else {
        // Ampliar la imagen en la posición del clic, usando porcentajes
        modalImg.style.transform = "scale(2)";
        modalImg.style.transformOrigin = `${xPercent}% ${yPercent}%`; // Zoom en la posición del clic
        modalImg.style.cursor = "zoom-out";
        modalImg.style.width = "1300px";
        isZoomed = true;
    }
}


// Cerrar modal al hacer clic fuera de la imagen
window.onclick = function (event) {
    if (event.target == modal) {
        modal.style.display = "none";
        document.body.classList.remove('modal-open');
    }
}
document.getElementById('LimpiarButton1').addEventListener('click', function () {

    patientNameContainer.innerHTML = '';
    container.innerHTML = '';

    // Ocultar las radiografías
    const elementos = document.getElementsByClassName("card-image");
    for (let i = 0; i < elementos.length; i++) {
        elementos[i].style.display = "none";
    }
    // Recargar la página actual
    location.reload();

});




// Delegación de eventos para el select de "validar enfermedad"
document.addEventListener("change", function (event) {
    if (event.target && event.target.matches("select[id^='enfermedadVal-']")) {
        console.log("Select de validar enfermedad ha cambiado");

        const radiographyId = event.target.id.split('-')[1]; // Obtener el ID de radiografía
        const selectElement = document.getElementById(`enfermedadVal-${radiographyId}`);
        const otraEnfermedad = document.getElementById(`otraEnfermedad-${radiographyId}`);
        const enfermedadValidada = document.getElementById(`enfermedadValidada-${radiographyId}`);

        if (selectElement.value === "Otro") {
            // Muestra el input cuando se selecciona 'Otro'
            otraEnfermedad.style.display = "block";
            enfermedadValidada.value = "";
        } else {
            // Oculta el input cuando se selecciona cualquier otra opción
            otraEnfermedad.style.display = "none";
            enfermedadValidada.value = selectElement.value;
        }
    }
});

// Delegación de eventos para el botón "diagnosticar"
document.addEventListener("click", async function (event) {
    if (event.target && event.target.matches("button[id^='diagnosticar-']")) {
        console.log("Botón diagnosticar presionado");

        const radiographyId = event.target.getAttribute('data-radiography-id');
        const diagnostico = document.getElementById(`enfermedadValidada-${radiographyId}`).value;

        const formData = new FormData();
        formData.append('idRx', radiographyId);
        formData.append('diagnostico', diagnostico);

        try {
            const response = await fetch(`${baseUrl}/diagnostics/api/diagnosticar-by-idRx/`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                console.log('Diagnóstico asignado exitosamente:', data);
                alert('Diagnóstico realizado con éxito');
                actualizarRadiografia(radiographyId); // Llamada a la función para actualizar solo esta sección

            } else {
                console.error('Error en la asignación del diagnóstico:', data.error);
            }
        } catch (error) {
            console.error('Error en la solicitud:', error);
        }
    }
});
