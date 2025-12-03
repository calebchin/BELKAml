import { GoogleAuth } from "google-auth-library";
import express from "express";
import cors from "cors";
import fetch from "node-fetch";

// GoogleAuth auto-pulls credentials inside Cloud Run / Cloud Functions
const auth = new GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/cloud-platform"],
});
const client = await auth.getClient();
const token = await client.getAccessToken();

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
          Authorization: `Bearer ${token}`,
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
      bindingProbability: (await predict(molecule)).predictions[0][0],
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export const handler = app;
