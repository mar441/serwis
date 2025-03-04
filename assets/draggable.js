document.addEventListener('DOMContentLoaded', function () {
    const modal = document.querySelector('.draggable-modal .modal-content');
    let isDragging = false;
    let offsetX = 0;
    let offsetY = 0;

    modal.addEventListener('mousedown', function (e) {
        isDragging = true;
        offsetX = e.clientX - modal.getBoundingClientRect().left;
        offsetY = e.clientY - modal.getBoundingClientRect().top;
        modal.style.cursor = 'grabbing';
    });

    document.addEventListener('mousemove', function (e) {
        if (isDragging) {
            modal.style.left = e.clientX - offsetX + 'px';
            modal.style.top = e.clientY - offsetY + 'px';
        }
    });

    document.addEventListener('mouseup', function () {
        isDragging = false;
        modal.style.cursor = 'grab';
    });
});
