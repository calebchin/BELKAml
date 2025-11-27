import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();

// middleware
app.use(cors());
app.use(express.json());

app.post("/api/binding-probability", async (req, res) => {
  try {
    const { molecule, protein } = req.body;

    if (!molecule || !protein) {
      return res.status(400).json({
        error: "Missing required fields: molecule or protein",
      });
    }

    // const r = await fetch(process.env.MODEL_ENDPOINT, {
    //   method: "POST",
    //   headers: { "Content-Type": "application/json" },
    //   body: JSON.stringify(req.body),
    // });
    // const prediction = await r.json();

    res.json({
      molecule,
      protein,
      bindingProbability: Math.random(),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export const handler = app;
