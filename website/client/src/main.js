$(function () {
  $("#belkaml-form").validate({
    rules: {
      molecule: { required: true },
      protein: { required: true },
    },
    submitHandler: async function (form) {
      // This only executes when form is VALID
      const payload = {
        molecule: $("#molecule").val(),
        protein: $("#protein").val(),
      };

      $(form).find('button[type="submit"]').prop("disabled", true);

      // psuedo API call
      // setTimeout(function () {
      //   const bindingProbability = Math.random();
      //
      //   $("#molecule-display").text(payload.molecule);
      //   $("#protein-display").text(payload.protein);
      //   $("#binding-probability-display").text(bindingProbability.toFixed(4));
      //
      //   $("body").data("molecule", payload.molecule);
      //   $("body").data("protein", payload.protein);
      //   $("body").data("binding-probability", bindingProbability);
      //
      //   $(form).find('button[type="submit"]').prop("disabled", false);
      //   $(".p1").addClass("hidden");
      //   $(".p2").removeClass("hidden");
      // }, 1500);

      try {
        const res = await fetch("http://localhost:8080/api/binding-probability", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });
        const { molecule, protein, bindingProbability } = await res.json();

        $("#molecule-display").text(molecule);
        $("#protein-display").text(protein);
        $("#binding-probability-display").text(bindingProbability.toFixed(4));

        $("body").data("molecule", molecule);
        $("body").data("protein", protein);
        $("body").data("binding-probability", bindingProbability);

        $(form).find('button[type="submit"]').prop("disabled", false);
        $(".p1").addClass("hidden");
        $(".p2").removeClass("hidden");
      } catch (err) {
        console.log(err);
      }

      return false; // prevent default form submission
    },
  });

  $("#btn-back").on("click tap", function (event) {
    $("#belkaml-form ")[0].reset();

    $(".p1").removeClass("hidden");
    $(".p2").addClass("hidden");
  });
});
