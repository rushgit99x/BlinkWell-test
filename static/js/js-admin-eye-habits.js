document.addEventListener('DOMContentLoaded',()=>{
	const flashes=document.querySelectorAll('.flash');
	flashes.forEach(f=>setTimeout(()=>f.remove(),4000));
});
