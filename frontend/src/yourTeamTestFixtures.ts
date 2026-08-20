import type { ProjectionRow } from "./data";
import { optimizerPlayersFromProjections, type OptimizerPlayer } from "./yourTeamOptimizer";

const quotas = { GKP: 2, DEF: 5, MID: 5, FWD: 3 } as const;

export function projectionPool(extras = 2): ProjectionRow[] {
  const rows: ProjectionRow[] = [];
  let index = 0;
  let baseIndex = 0;
  for (const [position, count] of Object.entries(quotas)) {
    for (let positionIndex = 0; positionIndex < count + extras; positionIndex += 1) {
      const expected = 8 - positionIndex * 0.45 - index * 0.01;
      const appearance = Math.max(0.55, 0.98 - positionIndex * 0.025);
      rows.push({
        schema_version: "phase9_frontend_v1",
        season: "2026-27",
        gameweek: "1",
        stable_player_id: `${position}_${String(positionIndex).padStart(2, "0")}`,
        player: `${position} Player ${positionIndex}`,
        team:
          positionIndex < count
            ? `team_${baseIndex++ % 5}`
            : `extra_team_${position}_${positionIndex}`,
        position,
        price_tenths: String(45 + positionIndex),
        opponent_display: "ARS (H)",
        expected_points: String(expected * appearance),
        expected_points_given_appearance: String(expected),
        expected_minutes: String(appearance * 80),
        p_appearance: String(appearance),
        p_start: String(Math.max(0, appearance - 0.1)),
        prob_points_ge_5: String(Math.min(0.9, expected / 10)),
        status: "a",
        model_variant: "X2_TEST",
      });
      index += 1;
    }
  }
  return rows;
}

export function legalSquad(pool = projectionPool()): OptimizerPlayer[] {
  const players = optimizerPlayersFromProjections(pool);
  return Object.entries(quotas).flatMap(([position, count]) =>
    players.filter((player) => player.position === position).slice(0, count),
  );
}

export function changedPlayer(
  players: OptimizerPlayer[],
  id: string,
  changes: Partial<Pick<OptimizerPlayer, "expectedPoints" | "conditionalPoints" | "appearanceProbability" | "priceTenths" | "team">>,
): OptimizerPlayer[] {
  return players.map((player) => (player.id === id ? { ...player, ...changes } : player));
}
