import pandas as pd

# List of new files to inspect
files_to_inspect = [
    "akk_logic_patterns.csv",
    "akkadian_function_words.csv",
    "akkadian_grammar_engine.csv",
    "akkadian_morphology.csv",
    "akkadian_verbs.csv",
    "ASL_OSL_format.csv",
    "atf_conventions..csv",
    "atf_parsing_logic.csv",
    "borger.csv",
    "categories1.csv"
]

for filename in files_to_inspect:
    try:
        df = pd.read_csv(filename)
        print(f"--- {filename} ---")
        print(df.columns.tolist())
        print(df.head(2))
        print("\n")
    except Exception as e:
        print(f"Error reading {filename}: {e}\n")

Plaintext
--- akk_logic_patterns.csv ---
['Category', 'Subcategory', 'Logic_Rule', 'Pattern_Change', 'Example_Form', 'Derivation_Note']
    Category   Subcategory     Logic_Rule                    Pattern_Change Example_Form         Derivation_Note
0  Phonology    I_w_G_stem  Initial_w_G_Š  Vowel before w changes/lengthens        uššab  From *iwaššab (dwells)
1  Phonology  I_w_N_D_stem  Initial_w_N_D   w behaves like strong consonant      uwaššib        D-stem (settled)


--- akkadian_function_words.csv ---
['Lemma', 'Type', 'Meaning', 'Case_Governed', 'Person_Gender_Number', 'Notes']
  Lemma         Type      Meaning Case_Governed Person_Gender_Number Notes
0   adi  Preposition        until      Genitive                    -     -
1   ana  Preposition  to, for, at      Genitive                    -     -


--- akkadian_grammar_engine.csv ---
['Rule_ID', 'Category', 'Gender', 'Number', 'Case', 'Suffix_Pattern', 'Function_Description', 'Example_Word']
  Rule_ID        Category     Gender    Number        Case Suffix_Pattern       Function_Description     Example_Word
0  DEC_01  Noun/Adjective  Masculine  Singular  Nominative            -um             Subject (Doer)  kalbum / damqum
1  DEC_02  Noun/Adjective  Masculine  Singular  Accusative            -am  Direct Object (Recipient)  kalbam / damqam


--- akkadian_morphology.csv ---
['Category', 'Sub_Category', 'Input_Condition', 'Transformation_Rule', 'Example_Input', 'Example_Output']
               Category Sub_Category       Input_Condition                            Transformation_Rule Example_Input Example_Output
0  Possession_Construct    Masc_Sing   Ends in 1 consonant                   No change (drop case ending)      našpakum         našpak
1  Possession_Construct    Masc_Sing  Ends in 2 consonants  Insert vowel matching root between consonants        kalbum          kalab


--- akkadian_verbs.csv ---
['Word', 'Definition', 'Notes', 'Category']
    Word     Definition                  Notes Category
0  abnum  stone, weight                      -     Noun
1   abum         father  irregular plural abbū     Noun


--- ASL_OSL_format.csv ---
['Tag', 'Category', 'Description', 'Status', 'Rule_Notes']
         Tag Category                                       Description     Status                  Rule_Notes
0  @signlist   Header          Defines the project name of the signlist  Mandatory     Must be the first line.
1   @project   Header  Specifies the project where the signlist resides   Optional  Can differ from @signlist.


--- atf_conventions..csv ---
['ASCII_ATF', 'Character', 'Unicode_Codepoint', 'CDLI_Strict']
  ASCII_ATF Character Unicode_Codepoint CDLI_Strict
0        sz         š             U+161         yes
1        SZ         Š             U+160         yes


--- atf_parsing_logic.csv ---
['Order', 'Concept', 'Regex_Pattern', 'Database_Action', 'Target_Field', 'Result_Value']
   Order                 Concept       Regex_Pattern                 Database_Action Target_Field    Result_Value
0      1         Unicode Cleanup    sz|s,|t,|s'|h,|j  Normalize string (sz->š, etc.)  unicode_val         Dynamic
1      2  Semantic Determinative  ^\{(?! \+)(.*?)\}$              Set Type & Subtype   token_type  det (semantic)


Error reading borger.csv: Error tokenizing data. C error: Expected 5 fields in line 98, saw 6


--- categories1.csv ---
['CATEGORY', 'Akkadian_Term', 'English_Translation', 'Description', 'Context', 'Relevance']
    CATEGORY Akkadian_Term English_Translation                                    Description                        Context Relevance
0  COMMODITY        kaspum              Silver   Universal exchange medium and value standard    Base of the monetary system  Critical
1  COMMODITY       hurāsum                Gold  Precious metal for high-density value storage  Investment asset for naruqqum   Extreme



Python
# Check borger.csv again with error handling
try:
    borger_df = pd.read_csv("borger.csv", on_bad_lines='skip', engine='python')
    print("Borger columns:", borger_df.columns.tolist())
    print(borger_df.head(2))
except Exception as e:
    print("Borger failed:", e)

# Check parsing logic patterns
try:
    parsing_df = pd.read_csv("atf_parsing_logic.csv")
    print("Parsing logic:", parsing_df[['Regex_Pattern', 'Result_Value']].head())
except:
    pass

Plaintext
Borger columns: ['Número de Borger', 'Valores de Borger', 'Firmar', 'Número Unicode', 'Nombre Unicode']
  Número de Borger Valores de Borger Firmar Número Unicode                              Nombre Unicode
0                1              COMO      𒀸        U+12038                     SIGNO CUNEIFORME CENIZA
1                2             AŠ.AŠ      𒐀        U+12400  SIGNO CUNEIFORME SIGNO NUMÉRICO DOS CENIZA
Parsing logic:         Regex_Pattern    Result_Value
0    sz|s,|t,|s'|h,|j         Dynamic
1  ^\{(?! \+)(.*?)\}$  det (semantic)
2       ^\{\+(.*?)\}$  det (phonetic)
3             \|.*?\|        compound
4                .*#$            TRUE
