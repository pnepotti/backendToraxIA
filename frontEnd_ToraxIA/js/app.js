document.addEventListener('DOMContentLoaded', function () {
    //const patientSelect = document.getElementById('patientName');
    const doctorSelect = document.getElementById('doctorName');
    //const patientDniInput = document.getElementById('patientDni');
    const doctorMatric = document.getElementById('doctorMatricula');

    const patientNameLabel = document.getElementById('patientNameLabel');
    const doctorMatriculaLabel = document.getElementById('doctorMatriculaLabel');
    const predictBtn = document.getElementById("predictButton");
    const reportButton = document.getElementById('reportButton');
    const reportModal = document.getElementById('reportModal');
    const closeModal = document.querySelector('.close');


    // Evento para abrir el modal cuando se hace clic en "Reporte"
    reportButton.addEventListener('click', function () {
        reportModal.style.display = 'block';
    });

    // Evento para cerrar el modal
    closeModal.addEventListener('click', function () {
        reportModal.style.display = 'none';
    });

    // Escuchar el cambio en el select del doctor
    doctorSelect.addEventListener('change', function () {
        doctorMatriculaLabel.style.display = "block";
        const selectedOption = this.options[this.selectedIndex];
        const selectedMatricula = selectedOption.getAttribute('data-matricula');
        doctorMatric.innerHTML = selectedMatricula;
    });

    document.getElementById('diagnostic-form').addEventListener('submit', async function (e) {
        e.preventDefault();

        predictBtn.disabled = true;
        predictBtn.classList.add("botonGris");

        // Mostrar mensaje de procesamiento
        document.getElementById('loading').style.display = 'block';
        document.getElementById('result').innerHTML = '';

        //Para probar
        document.getElementById('imagen').innerHTML = document.getElementById('imageInput').files[0];

        // Obtener los valores de los campos
        const patientName = document.getElementById('patientName').textContent;
        const patientDni = document.getElementById('patientDni').value;
        const doctorName = document.getElementById('doctorName').value;
        const doctorMatricula = document.getElementById('doctorMatricula').textContent;
        const imageInput = document.getElementById('imageInput').files[0];
        const descripcion = document.getElementById('descripcion').value;




        const allowedExtensions = /(\.jpg|\.jpeg|\.png)$/i;
        if (!allowedExtensions.exec(imageInput.name)) {
            alert('Por favor, selecciona un archivo de imagen válido (JPG, JPEG, PNG).');
            document.getElementById('loading').style.display = 'none';
            return;
        }


        // Validaciones básicas
        if (!imageInput || !patientName || !patientDni || !doctorName) {
            alert('Por favor, completa todos los campos y selecciona una imagen.');
            document.getElementById('loading').style.display = 'none';
            return;
        }

        // Crear un FormData para enviar la imagen y los datos
        const formData = new FormData();

        // Agregar los datos al FormData de forma individual
        formData.append('patientName', patientName);
        formData.append('patientDni', patientDni);
        formData.append('doctorName', doctorName);
        formData.append('doctorMatricula', doctorMatricula);
        formData.append('image', imageInput);
        formData.append('descripcion', descripcion);

        // Verifica que los datos sean correctos antes de enviarlos
        for (var pair of formData.entries()) {
            console.log(pair[0] + ', ' + pair[1]);
        }

        try {
            // Realizar la solicitud a la API
            const response = await fetch('http://localhost:8000/diagnostics/api/diagnostic/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error en la solicitud: ${response.status}`);
            }

            const data = await response.json();

            // Manejo de respuesta cuando no es una radiografía de tórax
            if (data.message) {
                document.getElementById('result').innerHTML = `Aviso: ${data.message}`;  // Mostrar que no es una radiografía de tórax
            } else {
                // Mostrar el resultado del diagnóstico

                document.getElementById('result').innerHTML = `
    <div><strong>ID de radiografía:</strong> ${data.radiography_id}</div>            
    <div><strong>Predicción de diagnóstico:</strong> ${data.prediccion}</div>
    <div><strong>Probabilidad:</strong> ${data.probability.toFixed(2)}</div>
    <div><strong>Confianza:</strong> ${data.confidence.toFixed(2)}</div>
    <div><strong>Incertidumbre:</strong> ${data.entropy.toFixed(2)}</div>

`;

                // Mostrar botón de reporte y llenar modal
                reportButton.style.display = 'block';// Obtener la fecha y hora actual
                const fechaActual = new Date();

                // Formatear la fecha y la hora por separado
                const fecha = fechaActual.toLocaleDateString(); // Solo la fecha
                const hora = fechaActual.toLocaleTimeString();   // Solo la hora

                document.getElementById('reportHospital').textContent = "EM405";
                document.getElementById("fecha").innerHTML = `<strong>Fecha y hora:</strong> ${fecha} ${hora}`;
                document.getElementById('reportRadiographyId').textContent = data.radiography_id;
                document.getElementById('reportPatient').textContent = patientName;
                document.getElementById('reportPatientDni').textContent = patientDni;
                document.getElementById('reportDoctor').textContent = doctorName;
                document.getElementById('reportPrediccion').textContent = data.prediccion;
                document.getElementById('reportProbability').textContent = data.probability.toFixed(2);
                document.getElementById('reportConfidence').textContent = data.confidence.toFixed(2);
                document.getElementById('reportEntropy').textContent = data.entropy.toFixed(2);
                document.getElementById('reportDescripcion').textContent = descripcion;



                // Configurar la imagen en el reporte
                const imageUrl = document.getElementById('imagen').src;
                document.getElementById('reportImage').src = imageUrl;
            }
        } catch (error) {
            console.error('Error al enviar la imagen:', error);
            document.getElementById('result').innerHTML = 'Hubo un error al procesar la imagen.';
        } finally {
            // Ocultar el mensaje de carga
            document.getElementById('result').style.display = 'block';
            document.getElementById('loading').style.display = 'none';
            document.getElementById('imagen').style.display = 'block';
        }
    });

    // Función para imprimir el reporte
    window.printReport = function () {



        // Obtener el contenido del modal y el contenido original del body
        const printContents = `
        <div style="background-color: #e8e8e8; padding: 20px; border: 1px solid #888; width: 80%; max-width: 700px; margin: auto; text-align: center;">
            <h2>Informe de Radiología</h2>
            <div>
                <img src="${document.getElementById('reportImage').src}" alt="Radiografía de tórax" style="width: 100%; max-height: 500px; margin-bottom: 20px;">
            </div>
            <div style="text-align: left; margin: 0 auto; font-size: 14px; line-height: 1.6;">
                <p><strong>Hospital:</strong> ${document.getElementById('reportHospital').textContent}</p>
                <p id="fecha"><strong>${document.getElementById('fecha').textContent}</p>
                <p><strong>Paciente:</strong> ${document.getElementById('reportPatient').textContent}</p>
                <p><strong>DNI de paciente:</strong> ${document.getElementById('reportPatientDni').textContent}</p>
                <p><strong>ID de radiografía:</strong> ${document.getElementById('reportRadiographyId').textContent}</p>                
                <p><strong>Médico solicitante:</strong> ${document.getElementById('reportDoctor').textContent}</p>
                <p><strong>Predicción de diagnóstico:</strong> ${document.getElementById('reportPrediccion').textContent}</p>
                <p><strong>Probabilidad:</strong> ${document.getElementById('reportProbability').textContent}</p>
                <p><strong>Confianza:</strong> ${document.getElementById('reportConfidence').textContent}</p>
                <p><strong>Incertidumbre:</strong> ${document.getElementById('reportEntropy').textContent}</p>
                <p><strong>Descripción:</strong> ${document.getElementById('reportDescripcion').textContent}</p>
                <p><strong>Firma:</strong></p><br>
                    <p>Informe con prediccion de anomalías mediante inteligencia artificial</p>
                    <p>Nota: No es valido como diagnostico medico</p>
            </div>
        </div>
    `;

        const originalContents = document.body.innerHTML;

        // Reemplazar el contenido del body con el contenido del reporte
        document.body.innerHTML = printContents;

        // Ejecutar la impresión
        window.print();

        // Restaurar el contenido original del body
        document.body.innerHTML = originalContents;

        // Recargar la página para restaurar los eventos y la vista original
        window.location.reload();
    };

    document.getElementById('imageInput').addEventListener('change', function (event) {
        const file = event.target.files[0]; // Obtener el archivo seleccionado
        const patientDni = document.getElementById('patientDni').value; // Obtener el DNI del paciente ingresado

        // Comprobar si el nombre del archivo contiene el número de documento del paciente
        if (!file.name.includes(patientDni)) {
            alert(`La imagen debe ser del paciente con DNI: ${patientDni}`);
            document.getElementById('imagen').src = '';
            document.getElementById('imagen').style.display = 'none';
            document.getElementById('result').style.display = 'none';
            // Reiniciar la entrada de archivo para forzar al usuario a seleccionar otra imagen
            event.target.value = '';
            return;
        }

        // Eliminar la vista previa de la imagen anterior
        document.getElementById('imagen').src = '';
        document.getElementById('imagen').style.display = 'none';
        document.getElementById('result').style.display = 'none';

        if (file) {
            const reader = new FileReader();

            // Una vez que el archivo es leído, lo convertimos en una URL
            reader.onload = function (e) {
                // Asignamos el resultado (la URL de la imagen) al src de la etiqueta img
                document.getElementById('imagen').src = e.target.result;
                document.getElementById('imagen').style.display = 'block';
            };

            // Leer el archivo como una URL de datos
            reader.readAsDataURL(file);
        }
    });

    document.getElementById('LimpiarButton').addEventListener('click', function () {
        // Obtener el formulario
        var form = document.getElementById('diagnostic-form');
        predictBtn.disabled = false;
        predictBtn.classList.remove("botonGris");
        // Reinicializar el formulario
        form.reset();
        doctorMatriculaLabel.style.display = 'none';
        patientName.innerHTML = '';
        patientNameLabel.style.display = 'none';
        doctorMatric.innerHTML = '';
        // Eliminar la vista previa de la imagen
        document.getElementById('imagen').src = '';
        document.getElementById('imagen').style.display = 'none';
        document.getElementById('result').style.display = 'none';
        reportButton.style.display = 'none';
    });



    document.getElementById('buscarPatient').addEventListener('click', function () {

        const dniBuscar = document.getElementById('patientDni');
        const nameBuscarLabel = document.getElementById('patientName');
        patientNameLabel.style.display = "block";
        doctorMatric.style.display = "block";
        switch (dniBuscar.value) {
            case "35000000":
                nameBuscarLabel.innerHTML = "Juan Pérez";
                break;
            case "36000000":
                nameBuscarLabel.innerHTML = "Jorge Gómez";
                break;
            case "37000000":
                nameBuscarLabel.innerHTML = "Carlos Díaz";
                break;
            default:
                alert("El paciente no se encuentra en la base de datos");
        }
    });

});