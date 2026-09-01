"use strict";

let formTooltips = {};
let edgarLocations = {};
let divisionCodes = {};
let sicLabels = {};
let forms = [];
for(let form in secLookupData.submissionForms){
    let parts = secLookupData.submissionForms[form].split('|');
    forms.push({
        form: form,
        description: (parts[3]||'').trim() || (parts[2]||'').trim(),
        title: (parts[4]||'').trim()=='*'?'':(parts[4]||'').trim()
    });
}
forms = forms.sort(function (formA, formB) {
    if (formA.form > formB.form) {
        return 1;
    }
    if (formA.form < formB.form) {
        return -1;
    }
    return 0;
});

//mce 2020-09-28:  default control/hash values
const defaultFormSelections = {
    dateRange: '5y',
    locationType: "located",
    locationCode: "all",
    category: "all",
    page: "1"
};

let search_endpoint;

let $category,
    $forms;

$(document).ready(function() {
    $('#results-grid').find('*').not('#show-result-count, #hits').hide();
    var i;

    //configure company hints event on both keywords and company text boxes
    $('.entity')
        .keydown(function(event) { //capture tab on keydown as focus is lost before keyup can fire
            var code = event.keyCode || event.which;
            if(code==9 || code == 13) $('table.entity-hints tr.selected').click();
        })
        .keyup(function(event){ //not reliable to capture tab key as focus has left
            var code = event.keyCode || event.which;
            if(code == 38 || code == 40){  //up arrow or down arrow
                var currentSelectedIndex = $('table.entity-hints tr.selected').removeClass('selected').index(),
                    hintCount = $('table.entity-hints tr').length,
                    newSelectedIndex = Math.min(Math.max(currentSelectedIndex + (code==38?-1:1) ,0), hintCount-1);
                $('table.entity-hints tr:eq('+newSelectedIndex+')').addClass('selected');
                return;
            }
            if(code == 13){ //enter accepts the highlighted
                $('table.entity-hints tr.selected').click();
                hideCompanyHints();
                return;
            }
            getCompanyHints(this, $(this).val(), event.key);
        }).focusin(function(event){
        getCompanyHints(this, $(this).val(), event.key);
    }).focusout(function(event){
        hideCompanyHints();
    });

    $('#form-container').blur(hideCompanyHints);
    $('#form-container button, #form-container label, #form-container input:not(.entity)').focus(hideCompanyHints);

    //load form categories from the forms array
    for(i in forms){  //first create a hash table for form tooltips
        let toolTip = (forms[i].description && forms[i].description.length > 0) ? forms[i].description : forms[i].title;
        if(toolTip && toolTip.length > 0) formTooltips[forms[i].form] = {"toolTip" : toolTip};
        if (forms[i].title && forms[i].title.length > 0) formTooltips[forms[i].form].title = forms[i].title;
    }

    //load form categories
    var categoryOptions = ['<li class="dropdown-item list-item active" href="#" value="all" data="" tabindex="0" role="option">View all</li>'];
    var formCatTree = {};
    for(var c=0;c<formCategories.length;c++){
        categoryOptions.push('<li class="dropdown-item list-item" href="#" value="form-cat'  + (c) + '" data="'+ formCategories[c].forms.join(', ')+'" tabindex="0" role="option">'+formCategories[c].category + '</li>');
        for(var f in formCategories[c].forms) formCatTree[formCategories[c].forms[f]] = 'form-cat'+ c; //used to set classes in modal forms broswer
    }
    categoryOptions.push('<li class="dropdown-item list-item" href="#" value="custom" tabindex="0" role="option">Enter the filing types</li>');
    $('#category-type-grp ul').html(categoryOptions.join(''));

    $category = $('#category-select');
    $forms = $('#filing-types');
    $('.forms-input-group').hide(); //needs to be not hide on load to allow bootstrap to calculate width of input box
    $('#custom-forms-cancel').click(function(){
        $('.forms-input-group').slideUp(); //Hide filing types
        $('.form-category-group').slideDown(); //Show filing type category
        selectElementInCustomDropdown('#category-type-grp', 'all');
    });

    //also set the modal forms browser category selector, but leave out the 'exclude ownership' category
    categoryOptions.pop();  //custom option at end
    categoryOptions.splice(1,1);  //exclude the exclude insider transaction category
    $('#category-filter-grp ul').html(categoryOptions.join(''));
    var $checkForms = $('#forms-browser .modal-footer #check_all_forms');
    $checkForms.change(selectFormTypes);

    //Select or un-select all form types
    function selectFormTypes() {
        var value = $('#forms-browser #category-filter').attr('value');
        var $allCheckBoxes = $('#forms-browser .form-check');

        $allCheckBoxes = (value != 'all') ? $allCheckBoxes.filter('.'+value).find('input') : $allCheckBoxes.find('input');

        //Select or un-select all form types
        $checkForms.is(':checked') ? $allCheckBoxes.prop('checked', true) : $allCheckBoxes.prop('checked', false);
    }

    //create the modal browse form's checkboxes
    var checkBoxes = [];
    for(let i=0;i<forms.length;i++){
        let title = (formTooltips[forms[i].form] && formTooltips[forms[i].form].toolTip) ? formTooltips[forms[i].form].toolTip : "";
        checkBoxes.push('<div id="fcbd-'+ forms[i].form.replace(' ','_') +'" class="custom-control custom-checkbox filingtype-checkbox form-check '+(formCatTree[forms[i].form]||'')+'" alt="'+forms[i].description+'">\n' +
            '    <input type="checkbox" class="custom-control-input" id="fcb'+i+'">\n' +
            '    <label class="custom-control-label" for="fcb'+i+'">' +forms[i].form+'</label>\n' +
            '  </div>');
    }

    $('#forms-browser .modal-body').html(checkBoxes.join(''));
    //lastly, return the select form types and trigger a search on "filter"
    $('#custom_forms_set').click(function(){
        var selectedForms = [];
        $('#forms-browser .form-check input:checked').parent().find('label').each(function(){
            selectedForms.push($(this).text())
        });
        selectElementInCustomDropdown('#category-type-grp', selectedForms.length?'custom':'all', false);
        $('#filing-types').val(selectedForms.join(', '));
        setHashFromForm();  //trigger search
    });

    $('#forms-browser .modal-body input[type=checkbox]').keypress(function(event){
        var code = event.keyCode || event.which;

        if(code == 13) {  //Select checkbox when enter is pressed
            if ($(this).is(':checked') == false) {
                $(this).prop('checked', true);
            } else {
                $(this).prop('checked', false);
            }
        }
    });

    //set to forms category to custom when the forms field is edited
    $forms.keyup(function(){ $category.val('custom'); });

    $('#open-submission, #open-file').click(function(){
        $('#close-modal').click();
    });

    $('#forms-browser button.close, #forms-browser button.cancel').click(function(event) {
        clearFormTypes();
    });

    function clearFormTypes() {  //Clear all form types
        $('#forms-browser .form-check input:checked').prop('checked', false);
        $('.forms-input-group').slideUp(); //Hide filing types
        $('.form-category-group').slideDown(); //Show filing type category

        if ($('#forms-browser .modal-footer #check_all_forms').is(':checked') == true) $('#forms-browser .modal-footer #check_all_forms').prop('checked', false);
    }

    //load locations selector and configure selection xevent
    var $locations1 = $('#location-grp');
    $('#location-grp ul').append('<li class="dropdown-item list-item active" href="#" value="all" tabindex="0" role="option">View all</li>');
    for(i=0;i<secLookupData.locations.length;i++){
        $('#location-grp ul').append('<li class="dropdown-item list-item" href="#" value="'  + secLookupData.locations[i][0] +'" tabindex="0" role="option">'+ secLookupData.locations[i][1] + '</li>');
        edgarLocations[secLookupData.locations[i][0]] = secLookupData.locations[i][1];
    }

    $('#location-grp').on('shown.bs.dropdown', function(event) { //Scroll to the active element when the location drop-down is shown
        $('#location-grp ul').animate({scrollTop: Math.max($('#location-grp ul').scrollTop()+$('#location-grp li.active').position().top - $('#location-grp ul').height()/3, 0) });
    });

    function resetActiveElementInCustomDropDown($this) {//Resets the active element if the active and selected element are not same
        let $activeElement = $this.find('li.active'),
            selectedElement = $this.find('button span').attr('value');

        if ($activeElement.attr('value') != selectedElement) {
            $activeElement.removeClass('active');
            $this.find('li[value="' + selectedElement + '"]' ).addClass('active');
        }
    }

    $('.form-dropdown-group, #category-filter-grp').on('shown.bs.dropdown', function(event) { //Set the keyboard focus to active element when the drop-down is shown
        resetActiveElementInCustomDropDown($(this));
        $(this).find('li.active').focus();
    });

    $('.form-dropdown-group, #category-filter-grp').on('hidden.bs.dropdown', function(event) { //Reset the active element when a user closes the drop-down without selecting an element
        resetActiveElementInCustomDropDown($(this));
    });

    //Configure selection event for a custom drop-down
    $('.form-dropdown-group li.dropdown-item, #category-filter-grp li.dropdown-item').keydown(function(event) {
        var code = event.keyCode || event.which;

        if(code == 38 || code == 40){  //up arrow|down arrow key
            $(this).removeClass('active');
            (code ==38) ? $(this).prev().addClass('active') : $(this).next().addClass('active');
        } else if (code == 13) {
            $(this).click();
        }
    }).keyup(function(event) {
        var code = event.keyCode || event.which;

        if(code == 9 || code == 16){  //tab/shift+tab key
            $(this).closest('ul').find('li.active').removeClass('active');
            $(this).addClass('active');
        }
    }).mouseover(function(event) {
        $(this).closest('ul').find('li.active').removeClass('active');
        $(this).addClass('active').focus();
    });

    $('.form-dropdown-group li.dropdown-item').click(function(event){
        $(this).closest('ul').find('li.active').removeClass('active');
        $(this).closest('.form-dropdown-group').find('button span').html($(this).text()).attr({'value': $(this).attr("value"), 'data': $(this).attr("data")}).addClass('active');
        if($(this).closest('.form-dropdown-group').find('button').attr("id") == 'category-select') categoryChanged();
        setHashFromForm();
    });

    $('#category-filter-grp li.dropdown-item').click(function(event){
        $(this).closest('ul').find('li.active').removeClass('active');
        $(this).closest('#category-filter-grp').find('button span').html($(this).text()).attr({'value': $(this).attr("value"), 'data': $(this).attr("data")}).addClass('active');

        var value = $(this).attr('value');
        var $allCheckBoxes = $('#forms-browser .form-check');
        if(value=='all'){
            $allCheckBoxes.find('input').prop('checked', false);
        } else {
            $allCheckBoxes.hide().find('input').prop('checked', false);
            $allCheckBoxes = $allCheckBoxes.filter('.'+value);        // Filter checkboxes based on the selected category filter
        }

        $allCheckBoxes.show();
        selectFormTypes();
    });

    $('#show-filing-types').keypress(function(event){
        var code = event.keyCode || event.which;

        if(code == 13) {  //Show modal when enter is pressed on "browse filing types"
            $(this).click();
        }
    }).click(function(){
        $('.form-category-group').slideUp();
        $('.forms-input-group').slideDown();

        //Get the selected filing category
        var filingCategory = $('.form-category-group #category-select span').attr('value');
        var $modalFilingCategory = $('#forms-browser #category-filter-grp');

        //Check if the filing type category exists in the modal
        var hasCategoryInModal = $('#forms-browser #category-filter-grp li[value=' + filingCategory +']').html() != undefined;
        if(!hasCategoryInModal)  filingCategory = 'all'; //Set the filing category to 'all' to display all types in the modal when a category doesn't exist

        //Change the selected filing category in the modal when the category is changed
        if (filingCategory != $modalFilingCategory.find('span').attr("value")) {
            $modalFilingCategory.find('li[value="' + filingCategory + '"]').click();
        }

        //Show "browse filing types" modal
        $('#browse-filing-types').click();
    });

    //show modal on "browse filing types" click
    $('#browse-filing-types').keypress(function(event){
        var code = event.keyCode || event.which;
        //Show modal when enter is pressed on "browse filing types"
        if(code == 13) { $(this).click(); }
    }).click(function(){
        var formsBrowserModal = new bootstrap.Modal(document.getElementById("forms-browser"));
        formsBrowserModal.show();
    });

    //Focus on Modal when shown
    $('#forms-browser').on('shown.bs.modal', function() {
        $(this).find('#category-filter-btn').focus();
    });

    //configure date range selector and data pickers
    var $dateRangeSelect = $('#date-range-select').change(setDatesFromRange);
    $('#date-from, #date-to').datepicker({
        changeMonth: true,
        changeYear: true,
        minDate: "2001-01-01",
        maxDate: new Date(),
        dateFormat: 'yy-mm-dd',
        yearRange: '2001:' + new Date().getFullYear().toString()
    });

    $('#date-from, #date-to').change(function(){
        var toDate = getDateRangeSelectValue('#date-to', new Date());
        var fromDate = getDateRangeSelectValue('#date-from', "2001-01-01");

        //ensure from >= to
        if(fromDate>toDate){
            if($(this).attr('id')=='date-to'){
                $('#date-from').val(formatDate(toDate));
            } else {
                $('#date-to').val(formatDate(fromDate));
            }
        }

        $dateRangeSelect.val('custom');
        setHashFromForm();
    });

    $('#show-full-search-form').click(function(evt){
        showFullForm();
        $('#form-container #keywords').focus();
        evt.stopPropagation();
    });

    $('#show-short-search-form').click(function(evt){
        evt.preventDefault();
        $('#edgar-short-form').click();
        $('#form-container .entity:visible').focus();
    });

    //over-ride default form submit
    $('#form-container form').submit(function(evt){
        evt.preventDefault();

        //If short form and no entity chosen from the suggestions drop-down, the text in the entity input field will be full text keywords
        var $keywordsInput = $('#keywords'),
            $entityInput = $('#form-container .entity:visible'),
            regxCIK = /\(CIK \d{10}\)/,
            foundCIK = regxCIK.test($entityInput.val());

        if($keywordsInput.is(':hidden') && !foundCIK){
            $keywordsInput.val($entityInput.val());
            $entityInput.val('');
        }
        setHashFromForm();
    }); //intercept form submit events; also called programmatically

    // Display EDGAR Search short form
    $('#edgar-short-form').click(function(evt){
        //Check if filing types field is visible
        if ($('#filing-types:visible')) {
            $('.forms-input-group').slideUp(); //Hide the filing types category field
            $('.form-category-group').slideDown(); //Show the filing types category field
        }

        $('#clear').click(); //Reset the form
        $('#show-short-search-form').parent().removeClass('d-inline-block');
        $('#search_form .hide-on-short-form').addClass('d-none');
        $('#search_form .hide-on-full-form').show();
    });

    //clear button behavior
    $('#clear').click(function(evt){
        $('#results').hide();

        //allow default reset behavior to occur first
        setTimeout(function() {
            selectElementInCustomDropdown('#category-type-grp', "all", false);
            selectElementInCustomDropdown('#location-grp', "all", false);
            clearFormTypes();
            clearValidationErrors();
            setDatesFromRange();
            window.location.hash = ""; // Clear the params from URL
        }, 10);
    });

    //modal form events
    $('#highlight-previous').click(function(){showHighlight (-1)});
    $('#highlight-next').click(function(){showHighlight (1)});
    $('.location-type-option ').click(function(evt){
        evt.preventDefault(); //don't change the hash
        var newLocationType =  $(this).attr("id"),
            $this = $(this),
            $locationType = $('#location-type'),
            oldLocationType = $locationType.attr('data');
        if(newLocationType!=oldLocationType){
            $locationType .attr('data', this.id).html($this.html()); //change the location search type: business state vs incorporation state
            if($('#location-selected').attr('value')) setHashFromForm();
        }
    });

    $('#select-result-columns input[type=checkbox]').change(function(event){
        showResultColumnChanged($(this));
    });
    $('#search_form select').change(function(){setHashFromForm();});

    $('#results-pagination a').click(function(event) {
        event.preventDefault();
        if($(this).parent().hasClass('disabled')) {
            return;
        }
        var currentPage = parseInt($(this).parent().siblings('li.active').find('a').attr("data-value")),
            pageSelected = $(this).attr("data-value");
        $(this).parent().siblings('li.active').removeClass('active');

        if (pageSelected == 'nextPage' || pageSelected == 'previousPage') {
            pageSelected = (pageSelected == 'nextPage' ) ? (currentPage + 1) : (currentPage-1);
            $(this).closest('#results-pagination').find('a[data-value=' + pageSelected + ']').parent().addClass('active');
        } else {
            $(this).parent().addClass('active');
        }
        setHashFromForm(event, false, false);
    });

    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function (el) {
        new bootstrap.Tooltip(el);
    });


    $('#hits table.table th.filed').click(function(event) {
        if($(this).hasClass('sortable')){
            let hashObject = hashToObject(),
                nextSort = '';
            if(!hashObject.sort) {
                hashObject.sort = 'desc';
            } else if(hashObject.sort == 'desc'){
                hashObject.sort = 'asc';
            } else if(hashObject.sort == 'asc'){
                delete hashObject.sort;
            }
            let hashValues = [];
            for(let p in hashObject)
                if(hashObject[p]) hashValues.push(p+'='+encodeURIComponent(hashObject[p]));
                hasher.setHash(hashValues.join('&'));
        }
    });

    //hash event and initialization
    hasher.changed.add(loadFormFromHash); //update form on back/forward navigate
    hasher.changed.add(executeSearch); //execute search on hash change from form submit or pn navigate
    hasher.init();
    search_endpoint = getSearchEndpoint();
    hasher.initialized.add(loadFormFromHash); //populate form if values bookmarked/embedded in link
    hasher.initialized.add(executeSearch);//execute search on load if values bookmarked/embedded in link
});

function selectElementInCustomDropdown(groupId, enableElement, updateHash) {
    $(groupId + ' li.active').removeClass('active');
    let $selectedElement =  $(groupId + ' ul li[value=' + enableElement + ']').addClass('active');
    $(groupId + ' button span').html($selectedElement.html()).attr({'value' : $selectedElement.attr("value"), 'data': $selectedElement.attr("data")});
    if (updateHash == undefined) updateHash = true;
    if (updateHash) $('#category-type-grp li.active').click();
}

function showResultColumnChanged($this) {
    var $column = $('#hits .table .'+ $this.attr("value"));
    var isChecked =  $this.prop('checked');
    isChecked ? $column.removeClass('d-none') : $column.addClass('d-none'); //Show/Hide columns based on the change
}

/** Show/Hide the result columns */
function showResultColumns() {
    var $showResultColumns = $('#select-result-columns input[type=checkbox]');

    $showResultColumns.each(function(){
        var $column = $('#hits .table .'+ $(this).attr("value"));
        $(this).prop('checked') ? $column.removeClass('d-none') : $column.addClass('d-none');
    });
}

// Enables result columns checkbox
function enableResultColumns() {
    $('#select-result-columns input[type=checkbox]').prop('checked', false); // De-select all check boxes
    for(var i=0; i < defaultResultColumnsArray.length; i++) $('#select-result-columns input[type=checkbox]#col-' + defaultResultColumnsArray[i]).prop('checked', true); // Enable result columns checkbox

    if ($('#location-selected').attr('value') !== 'all') { //Enable located/incorporated check box if user selected a specific location from the drop down
        $('#select-result-columns input[type=checkbox]#col-' + $('#location-type').attr('data')).prop('checked', true);
    }
}

function getDateRangeSelectValue(elementId, defaultValue) {
    var formattedDate = null;
    var dateStr = $(elementId).val();

    if(dateStr == '') {
        formattedDate = formatDate(defaultValue);
        $(elementId).val(formattedDate);
    } else {
        formattedDate = new Date(dateStr);
    }

    return formattedDate;
}


function categoryChanged(event){
    //Display filing types input field when a user selects to enter the types in the category drop down
    if($category.find('span').attr('value') == 'custom') {
        $('.form-category-group').slideUp();
        $('.forms-input-group').slideDown(function(){$('#filing-types').focus()});
        $('#forms-browser #category-filter').val("all").trigger('change');
    } else {
        $forms.val($category.find('span').attr('data'));
    }
}

function hashToObject(hashString){
    var hashObject = {};
    if(!hashString) hashString = hasher.getHashAsArray().pop();
    if(hashString){
        var hashTupletes = hashString.split('&');
        for(var i=0;i<hashTupletes.length;i++){
            var hashPair =  hashTupletes[i].split('=');
            hashObject[hashPair[0]]  = decodeURIComponent(hashPair[1]);
            if(hashPair[0].indexOf('filter_')===0) hashObject[hashPair[0]] = hashObject[hashPair[0]].replace(/>/g, "&gt;").replace(/</g, "&lt;");
        }
    }
    return hashObject;
}
function setHashFromForm(e, skipFacetFilters, resetActivePage){
    let oldHash = hasher.getHash();
    if(e) e.preventDefault();  //don't refresh the page if called by submit event
    if (skipFacetFilters == undefined) skipFacetFilters = true;
    if (resetActivePage == undefined) resetActivePage = true;
    var searchForms = checkForms($('#filing-types').val()),
        entityName = $('.entity').val(),
        locationType = $('#location-type').attr('data'),
        locationCode = $('#location-selected').attr('value'),
        dateRange = $('#date-range-select').find(':checked').val(),
        category = $('#category-select span').attr('value'),
        items = ($('#items').val()||'').trim().replace(/[, ]+/g,','),
        fileType = ($('#file-type').val()||'').trim();
    //only hash customize values if master selection is custom
    let keywords = $('#keywords').val().trim();
    let formValues = {
        q: keywords,
        dateRange: dateRange==defaultFormSelections.dateRange?null:dateRange,
        category: category==defaultFormSelections.category?null:category,
        locationType: locationType==defaultFormSelections.locationType?null:locationType,
        locationCode: locationCode==defaultFormSelections.locationCode?null:locationCode,
        sics: $('#sic-select').val(),
        ciks: extractCIK(entityName),
        entityName: entityName,
        sort: keywords && (skipFacetFilters || resetActivePage) ? oldHash.sort : null,
        fileType: fileType,
        items: items
    };
    if(formValues.dateRange == 'custom'){
        formValues.startdt = formatDate($('#date-from').val());
        formValues.enddt = formatDate($('#date-to').val());
    }
    if(formValues.category == 'custom' && searchForms.length) {
        formValues.forms = searchForms.replace(/[;,+]+\s*/g, ',').toUpperCase();
    }
    let hashValues = [];
    for(let p in formValues)
        if(formValues[p]) hashValues.push(p+'='+encodeURIComponent(formValues[p]));

    if (!skipFacetFilters) {
        let $filters = $('.filters .filter-group');
        let filterValues = {
            ciks: $filters.find('.entity-filter-group button').attr('data-button') ? extractCIK($filters.find('.entity-filter-group button').attr('data-button')) : '',
            entityName: $filters.find('.entity-filter-group button').attr('data-button') ? $filters.find('.entity-filter-group button').attr('data-button') : '',
            forms: $filters.find('.formtype-filter-group button') ? $filters.find('.formtype-filter-group button').attr('data-button') : '',
            location: $filters.find('.location-filter-group button') ? $filters.find('.location-filter-group button').attr('data-button') : ''
        };

        for(let p in filterValues)
            if(filterValues[p]) hashValues.push('filter_' + p+'='+encodeURIComponent(filterValues[p]));
    } else {
        $('div.filters').hide(); //Hide facet filters
    }

    if (resetActivePage) {
        $('#results-pagination li.active').removeClass('active'); //Remove active class for the page link
        $('#results-pagination a[data-value=1]').parent().addClass('active'); //Set first page as active
    }
    let page = $('#results-pagination li.active a').attr('data-value');
    if(page && page != defaultFormSelections.page) hashValues.push('page='+ page);
    hasher.setHash(hashValues.join('&')); //this trigger the listener (= executeSearch)
    let statusMessage = $('#no-results-grid div:visible h4').html();
    if ((statusMessage  == 'Search has timed out. Please try your query again!' || statusMessage == 'An unexpected error occurred. Please try again later!')   && (oldHash == hasher.getHash())) executeSearch(oldHash, hasher.getHash())   //Trigger search when the hash not changed and previous search has timed out
    return formValues;

    function checkForms(formString){ //look for user entered forms that might be missing comma separator
        const hasComma = formString.indexOf(',')!=-1;
        const hasSpace = formString.trim().indexOf(' ')!=-1;
        let allForms = [];
        for(let f=0; f<forms.length; f++){
            allForms.push(forms[f].form);
        }
        allForms.sort(function(a,b){return b.length-a.length});  //longest number of chars to shortest
        let userForms = formString.trim().toUpperCase().replace(/\s+/g, ' ').split(',');
        for(let uf=0;uf<userForms.length;uf++){
            userForms[uf] = userForms[uf].trim();
            if(allForms.indexOf(userForms[uf].trim())==-1) { //user form is not legit
                let rootForm = hasRootForm(userForms[uf]);
                if(rootForm) {
                    userForms[uf] = rootForm;
                } else {
                    let parts = userForms[uf].trim().split(' ');  //look for missing commas
                    if(parts.length>1){
                        userForms[uf] = ''; //rebuild it
                        let previousMatched = false;
                        for(let p=0;p<parts.length;p++){
                            rootForm = hasRootForm(parts[p]);
                            if(rootForm) parts[p] = rootForm;
                            userForms[uf] += (p>0 && (hasRootForm(parts[p-1]) || rootForm)?', ':'') + parts[p];
                        }
                    }
                }
            }
        }
        return userForms.join(', ');
        function hasRootForm(form){
            if(allForms.indexOf(form)!=-1) return form;
            if(form.substr(-2)=='/A' && allForms.indexOf(form.substr(0, form.length-2))){
                return form;  //.substr(0, form.length-2);
            } else {
                return false;
            }
        }
    }
}

function loadFormFromHash(hash){
    const hashValues = hashToObject(hash);

    $('#keywords').val(hashValues.q || '');
    let dateRange = hashValues.dateRange ? hashValues.dateRange : ((hashValues.startdt || hashValues.enddt) ? 'custom' : '5y');
    $('#date-range-select').val(dateRange);
    if(hashValues.startdt) {
        $('#date-from').val(formatDate(hashValues.startdt));
    } else if (dateRange == 'custom') {
        $('#date-from').val(formatDate('2001-01-01'));
    }
    if(hashValues.enddt) $('#date-to').val(formatDate(hashValues.enddt));
    let category = hashValues.category || (hashValues.forms?'custom': defaultFormSelections.category);
    selectElementInCustomDropdown('#category-type-grp', category, false);
    selectElementInCustomDropdown('#location-grp', hashValues.locationCode || defaultFormSelections.locationCode, false);
    $('#location-type').html($('#'+(hashValues.locationType || defaultFormSelections.locationType)).html());
    $('.entity').val(hashValues.entityName||'');
    $('#filing-types').val(hashValues.forms ? hashValues.forms.replace(/,/g,', ') : '');
    $('#file-type').val(hashValues.fileType ? hashValues.fileType.trim() : '');
    $('#items').val(hashValues.items ? hashValues.items.replace(/,/g,', ') : '');
    $('#sic-select').val(hashValues.sics||'');
    loadFacetFilterFromHash(hashValues, 'entity', "filter_entityName");
    loadFacetFilterFromHash(hashValues, 'form', "filter_forms");
    loadFacetFilterFromHash(hashValues, 'location', "filter_location");
    if ($('.filters .filter-group button').length > 0)  $('div.filters').show(); //Show Facet filters
    onFacetFilterClick(); //Register click event for the facet filters(breadcrumbs)
    $('#results-pagination li.active').removeClass('active');
    $('#results-pagination li a[data-value=' + (hashValues.page||1) + ']').parent().addClass('active');
    setDatesFromRange();
}

function loadFacetFilterFromHash(hashValues, facetKey, filterName) {
    let filterKey, filterValue;
    filterKey = hashValues[filterName];
    if(filterKey) filterKey = filterKey.replace(/>/g, "&gt;").replace(/</g, "&lt;").replace(/"/g, "");
    filterValue = filterKey;

    if (facetKey == 'location') {
        facetKey = ( (hashValues.locationType || 'located') == 'located') ? 'biz_states' : 'inc_states';
        filterValue = edgarLocations[filterValue];
    }

    var facetInfo =  facetControls[facetKey],
        $filter = $('.filters .filter-group .' + facetInfo['filter_class']);

    if (filterValue) {
        $filter.html('<button type="button" class="btn filter" aria-label="Remove filter ' + filterValue +  '" data-button="'
            + filterKey  + '"><span class="filter-type">'
            + facetInfo['facet_name'] + ': </span>' + filterValue + '<i class="fa fa-times"></i></button>');
    } else {
        $filter.html('');
    }
}

function setDatesFromRange(){
    var startDate = new Date(),
        endDate = new Date();
    switch($('#date-range-select').val()){
        case '30d':
            startDate.setDate(endDate.getDate()-30);
            break;
        case "1y":
            startDate.setFullYear(endDate.getFullYear()-1);
            break;
        case "5y":
            startDate.setFullYear(endDate.getFullYear()-5);
            break;
        case "10y":
            startDate.setFullYear(endDate.getFullYear()-10);
            break;
        case "all":
            startDate = new Date('2001-01-01');
            break;
        case "custom":
            return;
    }
    $('#date-from').val(formatDate(startDate));
    $('#date-to').val(formatDate(endDate));
}

function formatDate(dateOrString){
    var dt = new Date(dateOrString),
        yyyy=dt.getUTCFullYear(),
        m=dt.getUTCMonth()+1,
        d=dt.getUTCDate();

    return yyyy+'-'+(m<10?'0'+m:m)+'-'+(d<10?'0'+d:d);
}

function toTitleCase(title){
    var w, word, words = title.split(' ');
    for(w=0; w<words.length; w++){
        word = words[w].substr(0,1).toUpperCase() + words[w].substr(1).toLowerCase();
        if(w!=0 && ['A','An','And','In','Of','The'].indexOf(word)!=-1) word = word.toLowerCase();
        words[w] = word;
    }
    return words.join(' ');
}

function extractCIK(entityName){
    if(entityName.substr(entityName.length-16).match(/\(CIK [0-9]{10}\)/)
        || entityName.substr(entityName.length-12).match(/\([0-9]{10}\)/)
    )
        return entityName.substr(entityName.length-11, 10);
    if(!isNaN(entityName)&&parseInt(entityName).toString()==entityName&&parseInt(entityName)<1500)
        return parseInt(entityName);
    return false;
}

function getCompanyHints(control, keysTyped, keyPressed){
    if(keyPressed!='Enter' && keyPressed!='Escape' && keysTyped && keysTyped.trim()){
        const label = 'ajax call to ' + search_endpoint + ' for suggestion for "' + keysTyped + '"';
        var start = new Date();
        $.ajax({
            data: {keysTyped: keysTyped},
            dataType: 'JSON',
            type: 'GET',
            url: search_endpoint,
            success: function (data, status) {
                let end = new Date();
                //console.log('round-trip hint for ' + keysTyped + ' from ' + search_endpoint + ' = '+(Date.now()-start.getTime())+' ms');
                var hints = data.hits ? data.hits.hits : [];
                var hintDivs = [];
                if($('#form-container .entity:visible').val()==keysTyped){
                    var rgxKeysTyped = new RegExp('('+keysTyped.trim()+')','gi');
                    if(hints.length) {
                        for (var h = 0; h < hints.length; h++) {
                            const CIK = hints[h]._id,
                                entityName = hints[h]._source.entity;
                            hintDivs.push('<tr class="hint" data="' + entityName + ' (CIK ' + formatCIK(CIK) + ')"><td class="hint-entity">'
                                + (entityName || '').replace(rgxKeysTyped, '<b>$1</b>')
                                + '</td><td class="hint-cik">' + ((' <i>CIK ' + formatCIK(CIK).replace(rgxKeysTyped, '<b>$1</b>') + '</i>') || '') + '</td></tr>');
                        }

                        $('table.entity-hints').find('tr').remove()
                            .end().html(hintDivs.join('')).show().find('tr.hint')
                            .mousedown(function (evt) {
                                hintClicked(this)
                            })
                            .click(function (evt) {
                                hintClicked(this)
                            });
                        $('div.entity-hints').show();
                    } else {
                        hideCompanyHints();
                    }
                }
            },
            error: function (jqXHR, textStatus, errorThrown) {
                hideCompanyHints();
                console.log(textStatus);
                throw(errorThrown);
            }
        });
    } else {
        hideCompanyHints();
    }
}

function hintClicked(row){
    //console.log('hintClicked', Date.now());
    var $row = $(row);
    $('.entity').val($(row).attr('data'));
    hideCompanyHints();
    setHashFromForm();
}
function hideCompanyHints(){
    $('div.entity-hints').hide().find('tr').remove();
}
function showFullForm(){
    $('#search_form .hide-on-short-form').removeClass('d-none');
    $('#search_form .hide-on-full-form').hide();
    $('#show-short-search-form').parent().addClass('d-inline-block');
}

function clearValidationErrors() {
    if (! $('#search_form #date-validation-error').hasClass('d-none')) {
        $('#search_form #date-validation-error').addClass('d-none');
        $('#search_form .show-on-validation-error').removeClass('d-sm-block');
    }
}

function validateInput(searchParams) {
    if ('startdt' in searchParams && 'enddt' in searchParams) {
        var startDate = new Date(searchParams['startdt']);
        var endDate = new Date(searchParams['enddt']);
        if (isNaN(startDate) || isNaN(endDate) || (startDate > endDate)) {
            $('#search_form #date-validation-error').removeClass('d-none');
            $('#search_form .show-on-validation-error').addClass('d-sm-block');

            return false;
        }
    }

    return true;
}

function getSearchEndpoint() {
    let search_endpoint = `https://efts.sec.gov/LATEST/search-index`,
        hostname = location.hostname || '';

//START REMOVE FOR PROD!!!
    if(hostname.includes("www-test.sec.gov")) {
        search_endpoint = 'https://efts-stage.sec.gov/LATEST/search-index';  //STG
    } else if(!hostname.includes("sec.gov")) {
        search_endpoint = 'https://blxc6dsvd4.execute-api.us-east-1.amazonaws.com/LATEST/search-index'; //DEV
    }
//END REMOVE FOR PROD!!!

    return search_endpoint;
}

function executeSearch(newHash, oldHash) {
    clearValidationErrors();
    if (newHash == '') {
        if (!oldHash) return;  //blank form loaded
        if (oldHash != '') $('#results').hide();
        return; //Blank form loaded due to form reset
    }
    hideCompanyHints();
    var $searchingOverlay = $('.searching-overlay').show(),
        searchParams = hashToObject(newHash);

    if (searchParams.page) searchParams.from = (parseInt(searchParams.page || 1) - 1) * resultSize;
    if (!searchParams.forms) {
        if(searchParams.category && searchParams.category.substr(0,8) == 'form-cat'){
            let c = parseInt(searchParams.category.substr(8));
            if(c<=formCategories.length-1) searchParams.forms = formCategories[c].forms.join();
        }
    }
    //if(searchParams.ciks) searchParams.ciks = searchParams.ciks.split(',');
    if(searchParams.locationCode && searchParams.locationCode!='all') searchParams.locationCodes = searchParams.locationCode;
    if(!searchParams.startdt) searchParams.startdt = $('#date-from').val();
    if(!searchParams.enddt) searchParams.enddt = $('#date-to').val();
    if (!validateInput(searchParams)) {
        $searchingOverlay.hide();
        if(event) event.preventDefault();
        showFullForm();
        return;
    }
    $('.tooltip').remove();
    $('#no-results-grid div, #show-result-count').hide();  // Hide the no results and result count message. Alerts should be hidden and then displayed in IE for JAWS to announce the alert.
    $.ajax({
        data: searchParams,
        dataType: 'JSON',
        type: 'GET',
        url: search_endpoint,
        success: function (data, status) {
            let isXSL = false;
            var hits = data.hits ? data.hits.hits : [],
                totalHits = (hits.length > 0) ? data.hits.total.value : 0,
                rows = [];

            if (hits.length > 0) {
                if ($('#results').hasClass('visually-hidden')) { $('#results').removeClass('visually-hidden');  $('#results').find('*').removeClass('visually-hidden');   $('#results-grid').find('*').not('#show-result-count, #hits').css('display', '');}
                for(var i=0;i<hits.length;i++){
                    try{
                        let pathParts = hits[i]._id.split(':'),
                            adsh = pathParts[0],
                            fileName = pathParts[1],
                            fileNameParts = fileName.split('.'),
                            fileNameMain = fileNameParts[0],
                            fileNameExt = fileNameParts[1],
                            fileType = hits[i]._source.file_type || hits[i]._source.file_description || fileName,
                            form =  hits[i]._source.form,
                            firstCik = parseInt(hits[i]._source.ciks[0]),

                            root_form = hits[i]._source.root_form || hits[i]._source.root_forms.join();
                        if(hits[i]._source.xsl && fileNameExt.toUpperCase()=='XML') {//this is an XML file with a transformation
                            fileName = hits[i]._source.xsl + '/' + fileNameMain + '.' + fileNameExt;
                            isXSL = true;
                        }
                        let url = `${domain}/Archives/edgar/data/${firstCik}/${adsh.replace(/-/g,'')}/${fileName}`;
                        let display_names = getEntitiesAndCIKs(hits[i]._source.display_names),
                            entity_names = display_names[0],
                            ciks = display_names[1],
                            file_num = (hits[i]._source.file_num || ''),
                            formTitle = '';

                        if (root_form != "" && root_form in formTooltips) {
                            if (formTooltips[root_form].title && formTooltips[root_form].title.length > 0) formTitle = " (" + formTooltips[root_form].title + ")";
                        }
                        rows.push(`<tr>
                            <td class="filetype"><a href="${url}" class="preview-file" data-adsh="${adsh}" data-file-name="${fileName}">${form}${formTitle} ${form==fileType?'':' '+ fileType}</a></td>
                            <td class="filed">${hits[i]._source.file_date}</td>
                            <td class="enddate">${(hits[i]._source.period_ending||'')}</td>
                            <td class="entity-name">${entity_names.join('<br> ')}</td>
                            <td class="cik" nowrap>${ciks.join('<br> ')}</td>
                            <td class="biz-location located"  nowrap>${hits[i]._source.biz_locations.join('<br>')}</td>
                            <td class="incorporated" nowrap>${hits[i]._source.inc_states.map((code)=>{return edgarLocations[code]||code;}).join('<br>')}</td>
                            <td class="file-num" nowrap>${( (file_num != '') ? (Array.isArray(file_num)? file_num.map(function(filenum) {return '<a href="'+domain+'/cgi-bin/browse-edgar/?filenum='+ filenum + '&action=getcompany">' + filenum+'</a>';}).join('<br>') : '<a href="'+domain+'/cgi-bin/browse-edgar/?filenum='+file_num+'&action=getcompany">' +file_num+'</a>') : '')}</td>
                            <td class="film-num" nowrap>${(hits[i]._source.film_num ?  (Array.isArray(hits[i]._source.film_num)? hits[i]._source.film_num.join('<br>') : hits[i]._source.film_num) :'')}</td>
                            </tr>`);
                    } catch (e) {
                        console.log("error processing search hit:", hits[i]);
                    }
                }

                const $tbody = $('#hits table tbody').html(rows.join(''));

                // Re-apply persisted visited classes after re-render
                applyVisitedClasses($tbody);

                // Wrap previewFile to mark visited on normal click (no navigation),
                // but allow modified clicks to navigate (so native :visited can apply)
                function previewFileWrapper(e) {
                const isModifiedClick = e.ctrlKey || e.metaKey || e.shiftKey || e.button === 1;
                if (isModifiedClick) return; // Let the browser handle navigation

                e.preventDefault();
                markVisited(this);
                return previewFile.call(this, e, isXSL);
                }

                $tbody.find('a.preview-file').on('click', previewFileWrapper);

                $('#results-grid').show(); // Display results
                showPaginator(searchParams.page, totalHits);
                $('#show-result-count').css('display','block'); //Show the results count message. Alerts should be hidden and then displayed in IE for JAWS to announce the alert.
                $('#no-results-grid, #no-results-grid div').hide(); //Hide the no results message
                if (document.hasFocus() && document.activeElement != null && document.activeElement.className == 'page-link') {
                    $('#hits tbody').find('td:first a').focus();
                }

                //write the data to the HTML containers
                writeFilters(data, 'inc_states_filter');
                writeFilters(data, 'biz_states_filter');
                writeFilters(data, 'form_filter');
                writeFilters(data, 'entity_filter');

                if(searchParams.q){ //sortable
                    let sortIndicator = '';
                    if(searchParams.sort=='desc') sortIndicator = htmlArrows.down;
                    if(searchParams.sort=='asc') sortIndicator = htmlArrows.up;
                    $('#hits table.table th.filed').addClass('sortable').html('Filed ' + sortIndicator);
                } else { //sorted by default
                    $('#hits table.table th.filed').html('Filed ' + htmlArrows.down).removeClass('sortable');
                }

                $('#facets .card-header a').addClass('collapsed'); //collapse the accordion of facets
                $('div.collapse.show').removeClass('show');  //collapse the accordion of facets
            } else {
                if ($('#results').hasClass('visually-hidden')) {$('#results').removeClass('visually-hidden'); $('#results').find('*').removeClass('visually-hidden'); $('#results-grid').find('*').not('#show-result-count, #hits').css('display', '');}
                $('#results-grid').hide(); // Hide the results grid
                $('#no-results-grid').show(); // Display the no results message
                let statusMessage = "No results found for your search!";

                if (data.errorType == 'TimeoutError' || (data.errorMessage && data.errorMessage.indexOf('timed out') != -1)) {
                    statusMessage= 'Search has timed out. Please try your query again!';
                } else if (data.errorMessage && data.errorMessage.trim().length > 0) {
                    statusMessage = "An unexpected error occurred. Please try again later!";
                } else if(data.error && (typeof data.error == "string") && data.error.indexOf("Blank search not valid.") != -1) {
                    statusMessage = "Blank search not valid. Please specify the entity, keyword, location, or filing type.";
                } else if(searchParams.dateRange == '5y' || $('#date-range-select').val() == '5y') {
                    statusMessage = "No results found for the past 5 years. Consider expanding the filed date range!";
                }

                $('#no-results-grid div h4').html(statusMessage);
                $('#no-results-grid div').css('display','block');
            }

            showFullForm();
            categoryChanged();
            $searchingOverlay.hide();
            enableResultColumns(); //Enable result column check boxes
            showResultColumns(); //Show result columns selected by default
            $(window).scrollTop(0); //Scroll to the top of the page
            $('#results').show();
        },
        error: function (jqXHR, textStatus, errorThrown) {
            console.log(textStatus);
            hideCompanyHints();
            $searchingOverlay.hide();
            throw(errorThrown);
        }
    });

    function showPaginator(currentPage, totalHits) {
        if (currentPage == undefined) currentPage = 1;
        let totalHitsStr = totalHits.toLocaleString('en-US');
        if(totalHits < resultSize) {
            $('#results-pagination').hide(); // Hide Pagination
        } else {
            $('#results-pagination').show();
            let totalPages = (Math.ceil(totalHits/resultSize) <= maxResultPages) ? Math.ceil(totalHits/resultSize) : maxResultPages;
            $('#results-pagination li').removeClass('d-none');
            for (let i=totalPages+1; i<=maxResultPages; i++) {
                $('#results-pagination a[data-value=' + i + ']').parent().addClass('d-none');
            }
            togglePageLinks($('#results-pagination a[data-value=previousPage]'), currentPage, 1);
            togglePageLinks($('#results-pagination a[data-value=nextPage]'), currentPage, totalPages);
        }
        (currentPage == 1) ? $('#show-result-count h5').html(totalHitsStr + " search results") : $('#show-result-count h5').html("Page " + currentPage + " of " + totalHitsStr + " search results (" + resultSize + " results per page)" );
    }

    function togglePageLinks($pageLink, currentPage, page) {
        if (currentPage == page) {
            $pageLink.parent().addClass('disabled');
            $pageLink.attr('tabindex', '-1');
        } else {
            $pageLink.parent().removeClass('disabled');
            $pageLink.removeAttr('tabindex');
        }
    }

    function getEntitiesAndCIKs(display_names) {
        var entity_names = [], ciks=[];
        const regxCIK = /\(CIK \d{10}\)/;
        var foundCIK = false;

        display_names.forEach(function(display_name) {
            foundCIK = regxCIK.test(display_name);
            if(foundCIK) {
                var entity_name = display_name.substring(0, display_name.length-17);
                entity_names.push(entity_name);
                ciks.push(display_name.substring(entity_name.length+2,display_name.length-1));
            } else {
                entity_names.push(display_name)
            }
        });

        return [entity_names, ciks];
    }

    function writeFilters(data, filterKey){
        var $facetSection = $('#' + filterKey),
            $filter = $facetSection.find('.facets');
        if(data.aggregations && data.aggregations[filterKey] && data.aggregations[filterKey].buckets && data.aggregations[filterKey].buckets.length>=1){
            var htmlFilters = [],
                dataFilters = data.aggregations[filterKey].buckets;
            for(var i=0; i<dataFilters.length; i++){
                var tooltip = false,
                    aggr_key = dataFilters[i].key.replace(/\s+/,' ').trim(),  //remove extra spaces
                    display = aggr_key,
                    hiddenKey = aggr_key;
                switch(filterKey){
                    case 'inc_states_filter':
                    case 'biz_states_filter':
                        display = edgarLocations[hiddenKey] || hiddenKey;
                        break;
                    case 'sic_filter':
                        display = hiddenKey + ' ' + sicLabels[hiddenKey];
                        break;
                    case 'form_filter':
                        const form = forms.find(f => f.form === display);
                        if(form?.title){
                            display += ' <i>'+ form.title+ '</i>';
                        } 
                        tooltip =  (hiddenKey in formTooltips) ? formTooltips[hiddenKey].toolTip : '';
                        break;

                }
                htmlFilters.push('<tr> <td ' +(tooltip?'  data-bs-toggle="tooltip" data-bs-placement="right" title="'+tooltip+'"':'')+'>'
                    + '<a href="#0" data-filter-key="'+hiddenKey+'" class="'+filterKey+' ml-sm-2"><span class="badge badge-secondary text-bg-secondary mr-2">' + dataFilters[i].doc_count + '</span><span class="link-text ps-2">'+ display+'</span></a></td></tr>')

            }
            $filter
                .html('<table class="table" facet="'+filterKey+'">' + htmlFilters.join('') + '</table>')
                .find('a').click(function(evt){
                if(evt) evt.preventDefault();
                var $a = $(this),
                    filterKey = $a.closest('.facet').attr('id'),
                    facetInfo = facetControls[filterKey.replace('_filter', '')],
                    dataFilterKey = $a.attr('data-filter-key'),
                    filterType = dataFilterKey,
                    locationType = $a.closest('.facet').attr('data-location-type');
                if(locationType) $('#location-type').html(toTitleCase(locationType));
                if (filterKey == 'inc_states_filter' || filterKey == 'biz_states_filter') {
                    filterType = edgarLocations[dataFilterKey];
                }
                $('div#' + filterKey + ' a').addClass('collapsed'); //Collapse the facet
                $('.filters .filter-group .' + facetInfo['filter_class']).html('<button class="btn filter" aria-labelledby="' + filterKey.replace('_filter', '') + '"-filters" data-button="'
                    + dataFilterKey  + '"><span class="filter-type">'
                    + facetInfo['facet_name'] + ': </span>' + filterType + '<i class="fa fa-times"></i></button>');

                $('div.filters').show(); //Show Facet filters
                onFacetFilterClick(); //Register click event for the facet filters(breadcrumbs)
                setHashFromForm(evt, false);  //update the hash and trigger a new search -> update results and facets
            });

            $filter.find('[data-bs-toggle="tooltip"]').tooltip({fallbackPlacement: ['top']});
            $facetSection.show();
        } else {
            $filter.html('');
            $facetSection.hide();
        }
    }
}

function onFacetFilterClick() {
    $('.filters .filter-group').find('button').click(function(evt) {
        $(this).remove(); //Remove facet filter
        let skipFilters = false;
        if ($('.filters .filter-group button').length == 0) skipFilters = true;
        setHashFromForm(evt, skipFilters);
    });
}

function formatCIK(unpaddedCIK){ //accept int or string and return string with padded zero
    return Array(10-unpaddedCIK.toString().length+1).join('0') + unpaddedCIK.toString();
}

function replaceChar(keyword) {
    if (keyword) keyword = keyword.replace(/[+\-]/g, ' ').replace(/\s{2,}/,' ').replace(/\$/g, '').trim(); //replace + and - from the search terms as they are not indexed, remove extra spaces and dollar sign
    return keyword;
}

function keywordStringToPhrases(keywords){
    if(!keywords || !keywords.length) return false;
    keywords = keywords.trim();
    var phrases = [], phrase = '', inQuotes = false;
    for(var i=0;i<keywords.length;i++){
        switch(keywords[i]){
            case '"':
                if(phrase.trim().length) {
                    phrases.push(phrase.trim());
                    if (phrase.trim() !=  replaceChar(phrase)) phrases.push(replaceChar(phrase));
                    phrase = '';
                }
                inQuotes = !inQuotes;
                break;
            case ' ':
                if(!inQuotes){
                    if(phrase.length) {
                        if(!(phrase=='OR' || phrase=="NOT")) {
                            phrases = phrases.concat(replaceChar(phrase).split(" "));
                        }
                        phrase = '';
                        break;
                    }
                } //else condition falls through to default
            default:
                if(phrase.length == 0 && keywords[i] == ' ') continue;
                phrase += keywords[i];
        }
    }
    if(phrase.trim().length) phrases = phrases.concat(replaceChar(phrase).split(" ")); //final word/phrase
    return phrases;
}

function escapeRegExp(str) {
    return str.replace(/[.*+\-?^${}()|[\]\\]/g, '\\$&');
}

function highlightMatchingPhrases(text, phrases, isMarkupLanguage){
    if(phrases){
        var rgxPhrase ;
        for(var i=0;i<phrases.length;i++) {
            if(isMarkupLanguage){
                rgxPhrase = new RegExp('\\b('+ escapeRegExp(phrases[i]) + ')(?![^<>]*>)(?:\\b|\\W)', 'gmi');
                text = text.replace(rgxPhrase, '<span class="sect-efts-search-match-khjdickkwg">$1</span>');
            } else {
                rgxPhrase = new RegExp('\\b('+ escapeRegExp(phrases[i]) + ')(?!(?!.*?&lt;|&gt;)&gt;)(?:\\b|\\W)', 'gmi');
                text = text.replace(rgxPhrase, '<span class="sect-efts-search-match-khjdickkwg">$1</span>');
            }
        }
    }
    return text;
}


function getRandomTestFile(fileName) {
    const testFiles = ["AFSfull.pdf", "exhibit991.pdf", "exhibit992.pdf", "19AnnualReport2.pdf", "swn062817form11k.pdf", "17a5iii2017a.pdf"];
    const maxFiles = testFiles.length;

    if (testFiles.indexOf(fileName) == -1) {
        fileName = testFiles[Math.floor(Math.random() * maxFiles)];
    }

    return fileName;
}

function previewFile(evt, isXSL){
    $('#preview-status').html('');
    // $('#previewer').modal();
    const previewerModal = new bootstrap.Modal(document.getElementById("previewer"));
    previewerModal.show();
    $('#previewer .modal-title').hide();
    $('#previewer .modal-body #document').html('<div id="preview-spinner" class="justify-content-center align-items-center text-center"'
        + 'style="padding-top:' + $('#previewer div.modal-body').height()/2 + 'px">' + '<div class="spinner-border text-warning"></div></div>');
    $('#preview-status').append('<span class="visually-hidden">Loading document preview</span>');
    $('#previewer .modal-header .close').focus();
    if(evt) evt.preventDefault();
    let $a = $(this),
        ciks = $a.closest('tr').find('.cik').text(),
        cik = parseInt(ciks.substring(4, 14)),
        fileType = $a.text(),
        adsh = $a.attr('data-adsh'),
        form = $a.closest('tr').find('.form').text(),
        fileDate = $a.closest('tr').find('.filed').text(),
        fileName = $a.attr('data-file-name'),
        ext = fileName.split('.').pop().toLowerCase(),
        submissionRoot = domain + '/Archives/edgar/data/' + cik + '/' + adsh.replace(/-/g,'') + '/' ,
        hashValues = hashToObject(),
        searchedKeywords = (hashValues.q || '').replaceAll('<','&lt;').replaceAll('>','&gt;'),
        keyPhrases = keywordStringToPhrases(searchedKeywords),
        hostname = location.hostname || '',
        isSECHost = hostname.endsWith('.sec.gov'),
        url = $a.prop('href');
    try {
        if (ext == 'pdf') {
            const fileDir = "../sample_pdfFiles/";
            let previewEndpoint = isSECHost ? (submissionRoot + fileName) : encodeURIComponent(fileDir + getRandomTestFile(fileName));
            let viewerParams = 'file=' + previewEndpoint + (keyPhrases? "#search=" + encodeURIComponent(keyPhrases.join('|')) + "&entireWord=true" : "");
            $('#previewer .modal-body #document').html('<iframe src="web/viewer.html?' + viewerParams + '" style="height:100%;width:100%"></iframe>');
            $('#previewer .modal-body').css('overflow', '');
            showFilePreviewModal(hashValues, keyPhrases, form, ext, fileType, fileDate, false);
        }else {
            let previewEndpoint = isSECHost ? (submissionRoot + fileName) : ('getWebDoc.php?f=' + encodeURIComponent(submissionRoot + fileName) + '&repoint=true');
            $('#previewer .modal-body #document').html('<iframe id="ipreviewer" style="height:100%;width:100%" onload=""></iframe>');
            let ipv = document.getElementById('ipreviewer');
            ipv.style.visibility = 'hidden'; // make hidden so modal does not render anything until extensions are checked
            $(ipv).on("load", function () { //triggers after src set and loaded
                let highlightCount = 0,
                    idoc = ipv.contentDocument;
                if (idoc.documentElement && idoc.documentElement.innerHTML) {
                    const highlightStyle = '<STYLE>span.sect-efts-search-match-khjdickkwg {background-color:yellow;padding:2px;box-shadow:rgba(0,0,0,0.7) 0 1px 4px;}</STYLE>';
                    var rgxHead = /<head>/i,
                        hasHead = rgxHead.test(idoc.documentElement.innerHTML);
                    switch (ext) {
                        case 'xml':
                            if (isXSL) {
                                // Existing XSLT handling - already renders as HTML
                                if (hasHead) {
                                    idoc.documentElement.innerHTML = highlightMatchingPhrases(idoc.documentElement.innerHTML.replace(rgxHead, '<head>' + highlightStyle), keyPhrases, true)
                                } else {
                                    idoc.documentElement.innerHTML = '<head>' + highlightStyle + '</head>' + highlightMatchingPhrases(idoc.documentElement.innerHTML, keyPhrases, true);
                                }
                            } else {
                                // Plain XML - serialize, escape, and write as HTML via blob
                                const serializer = new XMLSerializer();
                                const rawXML = serializer.serializeToString(idoc);

                                const escapedXML = rawXML
                                    .replace(/&/g, '&amp;')
                                    .replace(/</g, '&lt;')
                                    .replace(/>/g, '&gt;');

                                const highlightedContent = highlightMatchingPhrases(escapedXML, keyPhrases, false);

                                const htmlContent = `<!DOCTYPE html>
                                    <html>
                                    <head>
                                        ${highlightStyle}
                                        <style>
                                            body { 
                                                font-family: monospace; 
                                                white-space: pre-wrap; 
                                                word-wrap: break-word;
                                                padding: 10px;
                                            }
                                        </style>
                                    </head>
                                    <body>${highlightedContent}</body>
                                    </html>`;

                                const blob = new Blob([htmlContent], { type: 'text/html' });
                                const blobURL = URL.createObjectURL(blob);

                                // Revoke the previous blob URL if one exists to avoid memory leaks
                                if (ipv.dataset.blobUrl) URL.revokeObjectURL(ipv.dataset.blobUrl);
                                ipv.dataset.blobUrl = blobURL;

                                // Remove and re-add the load listener to avoid infinite loop
                                $(ipv).off('load');
                                $(ipv).attr('src', blobURL);

                                // Re-attach load event to handle highlight counting after blob loads
                                $(ipv).one('load', function() {
                                    let blobDoc = ipv.contentDocument;
                                    const matches = blobDoc.documentElement.innerHTML
                                        .match(/<span class="sect-efts-search-match-khjdickkwg">/gm);
                                    highlightCount = matches ? matches.length : false;

                                    showFilePreviewModal(hashValues, keyPhrases, form, ext, fileType, fileDate, highlightCount);
                                    $('#highlight-previous').focus();
                                    showHighlight(0);
                                    ipv.style.visibility = 'visible';
                                });

                                // Return early since showFilePreviewModal will be called in the one('load') above
                                return;
                            }
                            break;
                        case 'txt':
                            //idoc.documentElement.innerHTML = '<head>' + highlightStyle + '</head><pre style="white-space:pre-wrap">' + highlightMatchingPhrases(idoc.body.innerHTML.replace(/&/g, '&amp;').replace(/>/g, '&gt;').replace(/</g, '&lt;'), keyPhrases, false) + '</pre>';
                            break;
                        default:
                            idoc.documentElement.innerHTML = idoc.documentElement.innerHTML.replace(/&amp;/g, '&').replace(/&gt;/g, '>').replace(/&lt;/g, '<');
                            if (hasHead) {
                                idoc.documentElement.innerHTML = highlightMatchingPhrases(idoc.documentElement.innerHTML.replace(/<img ([^>]*)src="/gi, '<img $1 src="' + submissionRoot).replace(rgxHead, '<head>' + highlightStyle), keyPhrases, true)
                            } else {
                                idoc.documentElement.innerHTML = '<head>' + highlightStyle + '</head>' + highlightMatchingPhrases(idoc.documentElement.innerHTML.replace(/<img ([^>]*)src="/gi, '<img $1 src="' + submissionRoot), keyPhrases, true);
                            }
                    }
                    var matches = idoc.documentElement.innerHTML.match(/<span class="sect-efts-search-match-khjdickkwg">/gm);
                    highlightCount = matches ? matches.length : false;
                }
                showFilePreviewModal(hashValues, keyPhrases, form, ext, fileType, fileDate, highlightCount);
                $('#highlight-previous').focus();
                showHighlight(0);
                ipv.style.visibility = 'visible';
            });
            $(ipv).attr("src", previewEndpoint);
        }
        $('#open-file').attr('href', submissionRoot + fileName);
        $('#open-submission').attr('href',submissionRoot + adsh + '-index.html');
    } catch(e) {
        console.log(e);
    }
}

function showFilePreviewModal(hashValues, keyPhrases, form, ext, fileType, fileDate, showHighlightedTermCount, isRawXML) {
    if(keyPhrases && ['txt','xml','htm','html', 'pdf'].indexOf(ext)!=-1){
        $('#previewer h4.modal-title strong').html((hashValues.q || '').replaceAll('<','&lt;').replaceAll('>','&gt;'));
        $('#previewer .modal-file-name').html((form!=fileType?fileType+ ' of ':'') + form + ' filed ('+fileDate+')');
        // Enabled by default (Disabled for PDF Preview)
        if (showHighlightedTermCount) {
            $('#previewer h4.modal-title span.find-counter').html('<span id="showing-highlight">1</span> of '+ showHighlightedTermCount);
            $('#previewer h4.modal-title #find-counter-wrapper').removeClass('d-none');
        } else   $('#previewer h4.modal-title #find-counter-wrapper').addClass('d-none');
        $('#previewer h4.modal-title').show();
    }
}

function browserForms(){
    var formsBrowserModal = new bootstrap.Modal(document.getElementById("forms-browser"));
    formsBrowserModal.show();

}

function showHighlight (change){
    let ipv = document.getElementById('ipreviewer'),
        $previewPanel = $('#previewer div.modal-body'),
        $previewHeader = $('#previewer div.modal-header'),
        highlights,
        $highlights,
        newShowing;
    if(ipv){
        highlights = ipv ? ipv.contentDocument.querySelectorAll('span.sect-efts-search-match-khjdickkwg') : $('span.sect-efts-search-match-khjdickkwg');
        newShowing = Math.min(Math.max(1, parseInt($('#showing-highlight').html())+change), highlights.length);
        if(highlights.length > 0){
            $('#showing-highlight').html(newShowing);
            //$previewPanel.animate({scrollTop: Math.max(0, $previewPanel.scrollTop() + $(highlights[newShowing-1]).offset().top - $previewPanel.height()/2 - $(window).scrollTop() - $previewHeader.outerHeight())}, 500);
            ipv.contentWindow.scrollTo(0, Math.max(0, $(highlights[newShowing-1]).offset().top - $previewPanel.height()/2));
        } else {
            $('#previewer h4.modal-title #find-counter-wrapper').addClass('d-none');
        }
    } else {
        $highlights = $('span.sect-efts-search-match-khjdickkwg');
        newShowing = Math.min(Math.max(1, parseInt($('#showing-highlight').html())+change), $highlights.length);
        if($highlights.length > 0){
            $('#showing-highlight').html(newShowing);
            $previewPanel.animate({scrollTop: Math.max(0, $previewPanel.scrollTop() + $($highlights.get(newShowing-1)).offset().top - $previewPanel.height()/2 - $(window).scrollTop() - $previewHeader.outerHeight())}, 500);
        } else {
            $('#previewer h4.modal-title #find-counter-wrapper').addClass('d-none');
        }
    }
}

const facetControls = {
    inc_states: {'facet_name': 'Incorporated In', 'filter_class': 'location-filter-group'},
    biz_states: {'facet_name': 'Located In', 'filter_class': 'location-filter-group'},
    entity: {'facet_name': 'Entity', 'filter_class': 'entity-filter-group'},
    form: {'facet_name': 'Form Type', 'filter_class': 'formtype-filter-group'},
};

const domain = 'https://' + (location.host.endsWith('.sec.gov')?location.host:'www.sec.gov');
const body_re = new RegExp('<body.*?>([\\s\\S]*)<\/body>','im');
const text_re = new RegExp('<TEXT>([\\s\\S]*)<\/TEXT>','im');
const resultSize = 100;
const maxResultPages = 10;
const defaultResultColumnsArray = ["filetype", "filed", "enddate"];

const formCategories = [
    {"category": "&nbsp;&nbsp;Exclude insider equity awards, transactions, and ownership (Section 16 Reports)", "forms":["-3","-4","-5"]},
    {"category": "All annual, quarterly, and current reports", "forms":["1-K","1-SA","1-U","1-Z","1-Z-W","10-D","10-K","10-KT","10-Q","10-QT","11-K","11-KT","13F-HR","13F-NT","15-12B","15-12G","15-15D","15F-12B","15F-12G","15F-15D","18-K","20-F","24F-2NT","25","25-NSE","40-17F2","40-17G","40-F","6-K","8-K","8-K12G3","8-K15D5","ABS-15G","ABS-EE","ANNLRPT","DSTRBRPT","IRANNOTICE","N-30B-2","N-30D","N-CEN","N-CSR","N-CSRS","N-MFP","N-MFP1","N-MFP2","N-PX","N-Q","NPORT-EX","NSAR-A","NSAR-B","NSAR-U","NT 10-D","NT 10-K","NT 10-Q","NT 11-K","NT 20-F","QRTLYRPT","SD","SP 15D2"]},
    {"category": "Insider equity awards, transactions, and ownership (Section 16 Reports)", "forms":["3","4","5"]},
    {"category": "Beneficial ownership reports", "forms":["SC 13D","SCHEDULE 13D","SC 13G","SCHEDULE 13G"]},
    {"category": "Exempt offerings", "forms":["1-A","1-A POS","1-A-W","253G1","253G2","253G3","253G4","C","D","DOS"]},
    {"category": "Registration statements and prospectuses", "forms":["10-12B","10-12G","18-12B","20FR12B","20FR12G","40-24B2","40FR12B","40FR12G","424A","424B1","424B2","424B3","424B4","424B5","424B7","424B8","424H","425","485APOS","485BPOS","485BXT","487","497","497J","497K","8-A12B","8-A12G","AW","AW WD","DEL AM","DRS","F-1","F-10","F-10EF","F-10POS","F-3","F-3ASR","F-3D","F-3DPOS","F-3MEF","F-4","F-4 POS","F-4MEF","F-6","F-6 POS","F-6EF","F-7","F-7 POS","F-8","F-8 POS","F-80","F-80POS","F-9","F-9 POS","F-N","F-X","FWP","N-2","POS AM","POS EX","POS462B","POS462C","POSASR","RW","RW WD","S-1","S-11","S-11MEF","S-1MEF","S-20","S-3","S-3ASR","S-3D","S-3DPOS","S-3MEF","S-4","S-4 POS","S-4EF","S-4MEF","S-6","S-8","S-8 POS","S-B","S-BMEF","SF-1","SF-3","SUPPL","UNDER"]},
    {"category": "Filing review correspondence", "forms":["CORRESP","DOSLTR","DRSLTR","UPLOAD"]},
    {"category": "SEC orders and notices", "forms":["40-APP","CT ORDER","EFFECT","QUALIF","REVOKED"]},
    {"category": "Proxy materials", "forms":["ARS","DEF 14A","DEF 14C","DEFA14A","DEFA14C","DEFC14A","DEFC14C","DEFM14A","DEFM14C","DEFN14A","DEFR14A","DEFR14C","DFAN14A","DFRN14A","PRE 14A","PRE 14C","PREC14A","PREC14C","PREM14A","PREM14C","PREN14A","PRER14A","PRER14C","PRRN14A","PX14A6G","PX14A6N","SC 14N"]},
    {"category": "Tender offers and going private transactions", "forms":["CB","SC 13E1","SC 13E3","SC 14D9","SC 14F1","SC TO-C","SC TO-I","SC TO-T","SC13E4F","SC14D9C","SC14D9F","SC14D1F"]},
    {"category": "Trust indenture filings", "forms":["305B2","T-3"]}
];

const htmlArrows = {
    up:  '&uarr;', // '↑',
    down: '&darr;', // '↓'
};

// --- Simulated "visited" handling for preview-only clicks ---
const VISITED_KEY = 'previewFileVisited';

function getVisitedSet() {
  try { return new Set(JSON.parse(localStorage.getItem(VISITED_KEY)) || []); }
  catch { return new Set(); }
}

function saveVisitedSet(set) {
  localStorage.setItem(VISITED_KEY, JSON.stringify([...set]));
}

function makeVisitedId(el) {
  // Use the exact href so it maps to what the user would actually visit
  // Fallback to data attributes if href is unavailable
  return el.href || (el.dataset ? `${el.dataset.adsh}|${el.dataset.fileName}` : '');
}

function markVisited(el) {
  const set = getVisitedSet();
  const id = makeVisitedId(el);
  if (id) {
    set.add(id);
    saveVisitedSet(set);
  }
  el.classList.add('visited');
}

function applyVisitedClasses(ctx) {
  const set = getVisitedSet();
  $(ctx).find('a.preview-file').each(function () {
    if (set.has(makeVisitedId(this))) this.classList.add('visited');
  });
};
