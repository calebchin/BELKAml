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

    async function predict(smiles) {
      const project = "belkaml";
      const location = "northamerica-northeast2";
      const endpointId = "1730486166584557568";
      const url = `https://${location}-aiplatform.googleapis.com/v1/projects/${project}/locations/${location}/endpoints/${endpointId}:predict`;

      const body = {
        instances: [{ smiles: smiles }],
      };

      const res = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.GOOGLE_ACCESS_TOKEN}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Error ${res.status} ${res.statusText}: ${text}`);
      }

      return await res.json();
    }

    res.json({
      molecule,
      protein,
      bindingProbability: await predict(molecule),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export const handler = app;
