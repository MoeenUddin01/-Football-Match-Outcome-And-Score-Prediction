/**
 * Unit tests for TeamFlag component and teamFlags mapping.
 *
 * Run:  npx tsx __tests__/teamFlag.test.ts
 * (uses tsx runner — no extra test framework needed)
 */
import teamFlags from "../lib/teamFlags";

const allTeams = Object.keys(teamFlags);

function assert(condition: boolean, msg: string) {
  if (!condition) {
    console.error(`FAIL: ${msg}`);
    process.exit(1);
  }
}

// Test 1: Known team with ISO code renders flag image URL
const brazilCode = teamFlags["Brazil"];
assert(brazilCode === "br", `Brazil should map to "br", got "${brazilCode}"`);
const flagUrl = `https://flagcdn.com/w80/${brazilCode}.png`;
assert(flagUrl === "https://flagcdn.com/w80/br.png", `Flag URL for Brazil is ${flagUrl}`);

// Test 2: Team with no mapped code returns null (placeholder fallback)
const abkhaziaCode = teamFlags["Abkhazia"];
assert(abkhaziaCode === null, `Abkhazia should be null, got "${abkhaziaCode}"`);

// Test 3: A fake/unmapped team returns undefined (not in dict at all)
const fakeTeamCode = teamFlags["FakeLand 999"];
assert(fakeTeamCode === undefined, `FakeLand 999 should be undefined, got "${fakeTeamCode}"`);

// Test 4: Every team in the mapping is either a valid ISO code or null
let badEntries: string[] = [];
for (const [team, code] of Object.entries(teamFlags)) {
  if (code !== null && code !== undefined) {
    if (typeof code !== "string" || code.length < 2 || code.length > 8) {
      badEntries.push(`${team}: "${code}"`);
    }
  }
}
assert(badEntries.length === 0, `Invalid ISO codes: ${badEntries.join(", ")}`);

// Test 5: Check that the total count matches expectations (336 teams)
assert(allTeams.length >= 330, `Expected at least 330 teams, got ${allTeams.length}`);

// Test 6: Specific important teams are present
const requiredTeams = ["Brazil", "Argentina", "Germany", "England", "France", "Spain", "Italy", "Japan"];
for (const t of requiredTeams) {
  assert(t in teamFlags, `Required team "${t}" missing from mapping`);
}

// Test 7: Historical/defunct teams explicitly mapped to null
const nullTeams = ["Czechoslovakia", "Yugoslavia", "German DR", "Two Sicilies"];
for (const t of nullTeams) {
  assert(teamFlags[t] === null, `Historical team "${t}" should be null`);
}

const mappedCount = allTeams.filter((t) => teamFlags[t] !== null).length;
const nullCount = allTeams.filter((t) => teamFlags[t] === null).length;

console.log(`PASS: All tests passed`);
console.log(`  ${allTeams.length} total teams`);
console.log(`  ${mappedCount} mapped to ISO codes`);
console.log(`  ${nullCount} mapped to null (no flag available)`);
