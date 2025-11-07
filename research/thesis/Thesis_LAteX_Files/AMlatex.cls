%###################################################################################################
%--- AMlatex ---------------------------------------------------------------------------------------
%###################################################################################################
% Description:       Auto-generated class supplying all modules and styles of the AMlatex-project
%
% License:           Copyright (c) 2024 Technical University of Munich (https://www.tum.de/) This
%                    work may be distributed and/or modified under the conditions of the LaTeX
%                    Project Public License, either version 1.3 of this license or (at your option)
%                    any later version. The latest version of this license is in http://www.latex-
%                    project.org/lppl.txt and version 1.3 or later is part of all distributions of
%                    LaTeX version 2005/12/01 or later. This work has the LPPL maintenance status
%                    'maintained'. The current maintainer of this work is Michael Kreutz
%                    (m.kreutz@tum.de).
%
% Build-Date:        21.03.2024
% Build-Time:        15:13:11
% Git Branch:        (HEADdetachedat5c510dd)
% Git Commit Hash:   5c510dd (5c510dd740e91334aef27bb4e8161a99f7b46fce)
%---------------------------------------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}[1994/06/01]%
\ProvidesClass{AMlatex}[2024/03/21 AMlatex]%
%
%###################################################################################################
%--- OPTIONS ---------------------------------------------------------------------------------------
%###################################################################################################
% List of available options:
% --------------------------
% AMcore                         (module)
% AMfont                         (module)
% --> optHelvetica               (module-option)
% --> optArial                   (module-option)
% --> optCharter                 (module-option)
% --> optComputerModern          (module-option)
% --> optCharterMath             (module-option)
% --> optComputerModernMath      (module-option)
% AMlang                         (module)
% --> optGerman                  (module-option)
% --> optEnglish                 (module-option)
% AMaddress                      (module)
% AMcolor                        (module)
% --> optRGB                     (module-option)
% --> optCMYK                    (module-option)
% --> optGray                    (module-option)
% --> optMonochrome              (module-option)
% AMtikz                         (module)
% --> optTikzExternalize         (module-option)
% AMlogo                         (module)
% AMmath                         (module)
% --> optLeftEquations           (module-option)
% --> optCenterEquations         (module-option)
% --> optLimitMathFonts          (module-option)
% AMlayout                       (module)
% --> optAfive                   (module-option)
% --> optAfour                   (module-option)
% --> optAthree                  (module-option)
% --> optAtwo                    (module-option)
% --> optAone                    (module-option)
% --> optAzero                   (module-option)
% --> optBeamerClassicFormat     (module-option)
% --> optBeamerWideFormat        (module-option)
% --> optExzellenz               (module-option)
% AMbiblio                       (module)
% --> optBiber                   (module-option)
% --> optBibtex                  (module-option)
% --> optBibstyleNumeric         (module-option)
% --> optBibstyleAlphabetic      (module-option)
% --> optBibstyleAuthorYear      (module-option)
% AMnames                        (module)
% AMutils                        (module)
% --> optHideTodos               (module-option)
% --> optHideAnnotations         (module-option)
% AMref                          (module)
% --> optBlackRefs               (module-option)
% AMstudentexpose                (style)
% AMstudentthesiscover           (style)
% --> optCoverWithMarks          (style-option)
% --> optCoverWithBorders        (style-option)
% AMbulletin                     (style)
% --> optStudentBulletin         (style-option)
% --> optResearchBulletin        (style-option)
% --> optTeachingBulletin        (style-option)
% --> optBulletinCropPage        (style-option)
% AMexercise                     (style)
% AMdocument                     (style)
% AMbeamer                       (style)
% --> optLectureSlides           (style-option)
% --> optExerciseSlides          (style-option)
% AMposter                       (style)
% AMphdprepublication            (style)
% AMphdthesis                    (style)
% AMempty                        (style)
% AMstudentthesis                (style)
% AMdocumentation                (style)
% AMlecturescript                (style)
% AMphdexpose                    (style)
% AMletter                       (style)
% AMdisplayitem                  (style)
%
% Main class options: (for enabling/disabling modules and selecting styles)
% -------------------
% Modules:
\DeclareOption{AMcore}{\def\optAMcore{x}}%
\DeclareOption{AMfont}{\def\optAMfont{x}}%
\DeclareOption{AMlang}{\def\optAMlang{x}}%
\DeclareOption{AMaddress}{\def\optAMaddress{x}}%
\DeclareOption{AMcolor}{\def\optAMcolor{x}}%
\DeclareOption{AMtikz}{\def\optAMtikz{x}}%
\DeclareOption{AMlogo}{\def\optAMlogo{x}}%
\DeclareOption{AMmath}{\def\optAMmath{x}}%
\DeclareOption{AMlayout}{\def\optAMlayout{x}}%
\DeclareOption{AMbiblio}{\def\optAMbiblio{x}}%
\DeclareOption{AMnames}{\def\optAMnames{x}}%
\DeclareOption{AMutils}{\def\optAMutils{x}}%
\DeclareOption{AMref}{\def\optAMref{x}}%
%
% Styles: (+ passing global options to baseclass)
\DeclareOption{AMstudentexpose}{\def\optAMstudentexpose{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMstudentthesiscover}{\def\optAMstudentthesiscover{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMbulletin}{\def\optAMbulletin{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMexercise}{\def\optAMexercise{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMdocument}{\def\optAMdocument{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMbeamer}{\def\optAMbeamer{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}}%
\DeclareOption{AMposter}{\def\optAMposter{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMphdprepublication}{\def\optAMphdprepublication{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMphdthesis}{\def\optAMphdthesis{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}}%
\DeclareOption{AMempty}{\def\optAMempty{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMstudentthesis}{\def\optAMstudentthesis{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}}%
\DeclareOption{AMdocumentation}{\def\optAMdocumentation{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}}%
\DeclareOption{AMlecturescript}{\def\optAMlecturescript{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}}%
\DeclareOption{AMphdexpose}{\def\optAMphdexpose{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMletter}{\def\optAMletter{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
\DeclareOption{AMdisplayitem}{\def\optAMdisplayitem{x}\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}}%
%
% Module and style options:
% ----------------------------
\DeclareOption{optHelvetica}{\def\optHelvetica{x}}%
\DeclareOption{optArial}{\def\optArial{x}}%
\DeclareOption{optCharter}{\def\optCharter{x}}%
\DeclareOption{optComputerModern}{\def\optComputerModern{x}}%
\DeclareOption{optCharterMath}{\def\optCharterMath{x}}%
\DeclareOption{optComputerModernMath}{\def\optComputerModernMath{x}}%
\DeclareOption{optGerman}{\def\optGerman{x}}%
\DeclareOption{optEnglish}{\def\optEnglish{x}}%
\DeclareOption{optRGB}{\def\optRGB{x}}%
\DeclareOption{optCMYK}{\def\optCMYK{x}}%
\DeclareOption{optGray}{\def\optGray{x}}%
\DeclareOption{optMonochrome}{\def\optMonochrome{x}}%
\DeclareOption{optTikzExternalize}{\def\optTikzExternalize{x}}%
\DeclareOption{optLeftEquations}{\def\optLeftEquations{x}}%
\DeclareOption{optCenterEquations}{\def\optCenterEquations{x}}%
\DeclareOption{optLimitMathFonts}{\def\optLimitMathFonts{x}}%
\DeclareOption{optAfive}{\def\optAfive{x}}%
\DeclareOption{optAfour}{\def\optAfour{x}}%
\DeclareOption{optAthree}{\def\optAthree{x}}%
\DeclareOption{optAtwo}{\def\optAtwo{x}}%
\DeclareOption{optAone}{\def\optAone{x}}%
\DeclareOption{optAzero}{\def\optAzero{x}}%
\DeclareOption{optBeamerClassicFormat}{\def\optBeamerClassicFormat{x}}%
\DeclareOption{optBeamerWideFormat}{\def\optBeamerWideFormat{x}}%
\DeclareOption{optExzellenz}{\def\optExzellenz{x}}%
\DeclareOption{optBiber}{\def\optBiber{x}}%
\DeclareOption{optBibtex}{\def\optBibtex{x}}%
\DeclareOption{optBibstyleNumeric}{\def\optBibstyleNumeric{x}}%
\DeclareOption{optBibstyleAlphabetic}{\def\optBibstyleAlphabetic{x}}%
\DeclareOption{optBibstyleAuthorYear}{\def\optBibstyleAuthorYear{x}}%
\DeclareOption{optHideTodos}{\def\optHideTodos{x}}%
\DeclareOption{optHideAnnotations}{\def\optHideAnnotations{x}}%
\DeclareOption{optBlackRefs}{\def\optBlackRefs{x}}%
\DeclareOption{optCoverWithMarks}{\def\optCoverWithMarks{x}}%
\DeclareOption{optCoverWithBorders}{\def\optCoverWithBorders{x}}%
\DeclareOption{optStudentBulletin}{\def\optStudentBulletin{x}}%
\DeclareOption{optResearchBulletin}{\def\optResearchBulletin{x}}%
\DeclareOption{optTeachingBulletin}{\def\optTeachingBulletin{x}}%
\DeclareOption{optBulletinCropPage}{\def\optBulletinCropPage{x}}%
\DeclareOption{optLectureSlides}{\def\optLectureSlides{x}}%
\DeclareOption{optExerciseSlides}{\def\optExerciseSlides{x}}%
%
% End of option processing
% ------------------------
\ProcessOptions\relax% End of option declaration
%
% Processing module dependencies:
% -------------------------------
\ifdefined\optAMfont% Automatic loading dependencies of module AMfont
    \def\optAMcore{x}%
\else\fi%
\ifdefined\optAMlang% Automatic loading dependencies of module AMlang
    \def\optAMcore{x}%
\else\fi%
\ifdefined\optAMcolor% Automatic loading dependencies of module AMcolor
    \def\optAMcore{x}%
\else\fi%
\ifdefined\optAMtikz% Automatic loading dependencies of module AMtikz
    \def\optAMcore{x}%
\else\fi%
\ifdefined\optAMlogo% Automatic loading dependencies of module AMlogo
    \def\optAMcore{x}%
    \def\optAMfont{x}%
    \def\optAMcolor{x}%
    \def\optAMlang{x}%
    \def\optAMtikz{x}%
\else\fi%
\ifdefined\optAMmath% Automatic loading dependencies of module AMmath
    \def\optAMcore{x}%
\else\fi%
\ifdefined\optAMlayout% Automatic loading dependencies of module AMlayout
    \def\optAMcore{x}%
    \def\optAMcolor{x}%
    \def\optAMtikz{x}%
    \def\optAMlogo{x}%
\else\fi%
\ifdefined\optAMbiblio% Automatic loading dependencies of module AMbiblio
    \def\optAMcore{x}%
    \def\optAMlang{x}%
\else\fi%
\ifdefined\optAMnames% Automatic loading dependencies of module AMnames
    \def\optAMcore{x}%
\else\fi%
\ifdefined\optAMutils% Automatic loading dependencies of module AMutils
    \def\optAMcore{x}%
    \def\optAMlang{x}%
    \def\optAMcolor{x}%
\else\fi%
\ifdefined\optAMref% Automatic loading dependencies of module AMref
    \def\optAMcore{x}%
    \def\optAMlang{x}%
    \def\optAMcolor{x}%
\else\fi%
%
% Check number of selected styles (for 0: auto-select AMempty as style (without explicit module selection))
%                                 (for 1: select this style and load ALL modules)
%                                 (for >1: throw error)
\newcounter{SelectedStyles}% Stores the number of selected styles
\ifdefined\optAMstudentexpose\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMstudentthesiscover\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMbulletin\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMexercise\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMdocument\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMbeamer\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMposter\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMphdprepublication\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMphdthesis\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMempty\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMstudentthesis\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMdocumentation\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMlecturescript\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMphdexpose\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMletter\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifdefined\optAMdisplayitem\stepcounter{SelectedStyles}\else\fi% Increase the counter by one...
\ifnum\value{SelectedStyles}=0%
%   Load default style
    \def\optAMempty{x}%
\else\fi%
\ifnum\value{SelectedStyles}=1%
%   Load ALL modules
    \def\optAMcore{x}%
    \def\optAMfont{x}%
    \def\optAMlang{x}%
    \def\optAMaddress{x}%
    \def\optAMcolor{x}%
    \def\optAMtikz{x}%
    \def\optAMlogo{x}%
    \def\optAMmath{x}%
    \def\optAMlayout{x}%
    \def\optAMbiblio{x}%
    \def\optAMnames{x}%
    \def\optAMutils{x}%
    \def\optAMref{x}%
\else\fi%
\ifnum\value{SelectedStyles}>1%
%   Throw error
    \typeout{! Please select only ONE style.}%
\else\fi%
%
% Check number of selected modules (for 0: load ALL modules)
\newcounter{SelectedModules}% Stores the number of selected modules
\ifdefined\optAMcore\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMfont\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMlang\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMaddress\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMcolor\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMtikz\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMlogo\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMmath\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMlayout\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMbiblio\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMnames\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMutils\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifdefined\optAMref\stepcounter{SelectedModules}\else\fi% Increase the counter by one...
\ifnum\value{SelectedModules}=0%
%   Load ALL modules
    \def\optAMcore{x}%
    \def\optAMfont{x}%
    \def\optAMlang{x}%
    \def\optAMaddress{x}%
    \def\optAMcolor{x}%
    \def\optAMtikz{x}%
    \def\optAMlogo{x}%
    \def\optAMmath{x}%
    \def\optAMlayout{x}%
    \def\optAMbiblio{x}%
    \def\optAMnames{x}%
    \def\optAMutils{x}%
    \def\optAMref{x}%
\else\fi%
%
%
%###################################################################################################
%--- BASE CLASS LOADING ----------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMstudentexpose\LoadClass[11pt,a4paper,fleqn]{article}\else\fi%
\ifdefined\optAMstudentthesiscover\LoadClass[11pt,a4paper]{article}\else\fi%
\ifdefined\optAMbulletin\LoadClass[11pt,a4paper,landscape]{article}\else\fi%
\ifdefined\optAMexercise\LoadClass[11pt,a4paper,fleqn]{article}\else\fi%
\ifdefined\optAMdocument\LoadClass[11pt,a4paper,fleqn]{article}\else\fi%
\ifdefined\optAMbeamer\LoadClass[noamssymb,t,11pt]{beamer}\else\fi%
\ifdefined\optAMposter\LoadClass[11pt,a4paper,fleqn]{article}\else\fi%
\ifdefined\optAMphdprepublication\LoadClass[11pt,a4paper,fleqn]{article}\else\fi%
\ifdefined\optAMphdthesis\LoadClass[11pt,a4paper,twoside]{book}\else\fi%
\ifdefined\optAMempty\LoadClass[11pt,a4paper,fleqn]{article}\else\fi%
\ifdefined\optAMstudentthesis\LoadClass[11pt,a4paper,twoside,fleqn]{book}\else\fi%
\ifdefined\optAMdocumentation\LoadClass[11pt,a4paper,oneside,fleqn]{book}\else\fi%
\ifdefined\optAMlecturescript\LoadClass[12pt,a4paper,twoside]{book}\else\fi%
\ifdefined\optAMphdexpose\LoadClass[11pt,a4paper,fleqn]{article}\else\fi%
\ifdefined\optAMletter\LoadClass[11pt,a4paper,fleqn]{article}\else\fi%
\ifdefined\optAMdisplayitem\LoadClass[11pt,a4paper,landscape]{article}\else\fi%
%
%
%###################################################################################################
%--- BEGIN MODULE AMcore ---------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMcore% Start-check if module AMcore is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMcore.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
%
\PassOptionsToPackage{utf8}{inputenc}%  sets input encoding to UTF-8
\RequirePackage{inputenc}%
\RequirePackage{etoolbox}%              for parsing conditional expressions
\RequirePackage{xparse}%                allows advanced command definitions
\RequirePackage{silence}%               for suppressing warnings
%
%
%=========================
%--- Warning Filtering ---
%=========================
\RequirePackage{soul}% workaround to prevent soulpos (fancy underlininig) from loading soulutf8; soulutf8 is obsolete (2023-04-04)
%
%
%=================
%--- Debugging ---
%=================
% Printing misc information
\def\AMcoreLogInfo#1{\typeout{#1.}}%
%
% Printing warnings
\def\AMcoreLogWarning#1{\typeout{LaTeX Warning: #1 on input line \the\inputlineno.}}%
%
% Printing errors
\def\AMcoreLogError#1{\typeout{! #1.}}%
%
%
%==========================
%--- Control Structures ---
%==========================
\newcommand{\AMcoreRepetition}[2]{\newcount\Step\Step=0\loop\ifnum\Step<#1\advance \Step by 1 {#2}\repeat}%
%
%
%============
%--- Misc ---
%============
% Suppress warning produced after including special PDF's (only use this command, if it is actually defined (since pdftex 1.40.15))
\begingroup\expandafter\expandafter\expandafter\endgroup%
\expandafter\ifx\csname pdfsuppresswarningpagegroup\endcsname\relax%
\else\pdfsuppresswarningpagegroup=1\relax\fi%
%
% TUMlatex project website
\newcommand{\AMcoreProjectWebsite}{https://gitlab.lrz.de/AM/TUMlatex}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMcore.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if module AMcore is enabled
%###################################################################################################
%--- END MODULE AMcore -----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMfont ---------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMfont% Start-check if module AMfont is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMfont.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%===============================
%--- Default Option handling ---
%===============================
% Text: Use Helvetiva as default, if no font has been selected
% -----
\ifdefined\optHelvetica\else%
\ifdefined\optArial\else%
\ifdefined\optCharter\else%
\ifdefined\optComputerModern\else%
    \def\optHelvetica{x}%
\fi%
\fi%
\fi%
\fi%
%
% Math: Use Charter as default, if no font has been selected
% -----
\ifdefined\optCharterMath\else%
\ifdefined\optComputerModernMath\else%
    \def\optCharterMath{x}%
\fi%
\fi%
%
%
%================
%--- Packages ---
%================
\PassOptionsToPackage{T1}{fontenc}%                     Use T1 font encoding (necessary for correct hyphenation of german language)
\RequirePackage{fix-cm}%                                Over­rides the (orig­i­nal) latex re­stric­tions on the sizes at which CM and EC fonts may be used
\RequirePackage{fontenc}%
\RequirePackage{helvet}%                                Load Helvetica font (very similar to Arial and TUMNeueHelvetica)
\ifdefined\optArial%
    \RequirePackage{uarial}%                            Load URW Arial font
\fi%
\ifdefined\optComputerModernMath%
    % Attention: to keep the math font as Computer Modern (default), we MUST NOT load the mathdesign package
    \usepackage{charter}%                               Load Charter font for text only (math remains Computer Modern)
\else%
    \PassOptionsToPackage{expert,charter,cal=cmcal}{mathdesign}%  Load Charter font for text and math (option expert to prevent "fake small capitals - fsc") (option cal=cmcal to prevent overwriting of caligraphic symbols)
    \RequirePackage{mathdesign}%
\fi%
\ifdefined\optComputerModern%
    \renewcommand{\sfdefault}{cmss}%                    Set Computer Modern as default Sans Serif font
    \renewcommand{\rmdefault}{cmr}%                     Set Computer Modern as default Roman font
\else%
    \ifdefined\optHelvetica%
        \renewcommand{\sfdefault}{phv}%                 Set Helvetica as default Sans Serif font
    \fi%
    \ifdefined\optArial%
        \renewcommand{\sfdefault}{ua1}%                 Set Arial as default Sans Serif font
    \fi%
    \ifdefined\optCharter%
        \renewcommand{\sfdefault}{phv}%                 Set Helvetica as default Sans Serif font (Charter does not specify sans serif styles)
    \fi%
    \renewcommand{\rmdefault}{bch}%                     Set Charter as default Roman font (Helvetica and Arial do not specify roman styles)
\fi
\renewcommand{\ttdefault}{cmtt}%                        Set Computer Modern as default Typewriter font
\RequirePackage{anyfontsize}%                           Allow any font size by scaling
%
%
%================
%--- Commands ---
%================
% Select font default style
% -------------------------
\ifdefined\optHelvetica%
    \renewcommand{\familydefault}{\sfdefault}\sffamily%  use sans serif type as default
\fi%
\ifdefined\optArial%
    \renewcommand{\familydefault}{\sfdefault}\sffamily%  use sans serif type as default
\fi%
\ifdefined\optCharter%
    \renewcommand{\familydefault}{\rmdefault}\rmfamily%  use roman type as default
\fi%
\ifdefined\optComputerModern%
    \renewcommand{\familydefault}{\rmdefault}\rmfamily%  use roman type as default
\fi%
%
% Font sizes
% ----------
\providecommand{\@ptsize}{1}% define ptsize if not already defined
%
% Font size 6 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeVI}{%
    \renewcommand\tiny{\@setfontsize\tiny{3.09}{3.64}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{4.19}{4.91}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{4.90}{5.89}%
        \abovedisplayskip 4.32\p@ \@plus 1.10\p@ \@minus 2.36\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 0.88\p@%
        \belowdisplayshortskip 2.33\p@ \@plus 1.06\p@ \@minus 1.26\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 2.33\p@ \@plus 0.90\p@ \@minus 0.90\p@%
            \parsep 1.26\p@ \@plus 0.72\p@ \@minus 0.55\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{5.44}{6.64}%
        \abovedisplayskip 5.35\p@ \@plus 1.65\p@ \@minus 2.71\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 1.45\p@%
        \belowdisplayshortskip 2.97\p@ \@plus 1.53\p@ \@minus 1.45\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 3.39\p@ \@plus 1.26\p@ \@minus 1.60\p@%
            \parsep 1.70\p@ \@plus 0.90\p@ \@minus 0.55\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{5.99}{7.27}%
        \abovedisplayskip 6.00\p@ \@plus 1.45\p@ \@minus 3.26\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 1.65\p@%
        \belowdisplayshortskip 3.47\p@ \@plus 1.82\p@ \@minus 1.65\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{6.98}{8.50}}%
    \renewcommand\Large{\@setfontsize\Large{8.33}{10.54}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{9.82}{12.57}}%
    \renewcommand\huge{\@setfontsize\huge{11.80}{14.55}}%
    \renewcommand\Huge{\@setfontsize\Huge{13.71}{16.45}}%
    %
    \setlength\smallskipamount{1.65\p@ \@plus 4.61\p@ \@minus 7.94\p@}%
    \setlength\medskipamount{3.29\p@ \@plus 1.10\p@ \@minus 1.10\p@}%
    \setlength\bigskipamount{6.58\p@ \@plus 2.19\p@ \@minus 2.19\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{6.58\p@}%
        \setlength\headsep{0.15in}%
        \setlength\topskip{6.00\p@}%
        \setlength\footskip{0.21in}%
        \setlength\@tempdimb{199.45\p@}%
    \fi%
    %
    \setlength\marginparsep{18.89\p@}%
    \setlength\marginparpush{3.08\p@}%
    %
    \setlength\footnotesep{4.13\p@}%
    \setlength{\skip\footins}{5.42\p@ \@plus 2.19\p@ \@minus 1.10\p@}%
    %
    \setlength\floatsep{6.58\p@ \@plus 1.10\p@ \@minus 1.43\p@}%
    \setlength\textfloatsep{10.97\p@ \@plus 1.10\p@ \@minus 2.19\p@}%
    \setlength\intextsep{6.92\p@ \@plus 1.43\p@ \@minus 1.43\p@}%
    %
    \setlength\dblfloatsep{6.92\p@ \@plus 1.10\p@ \@minus 1.43\p@}%
    \setlength\dbltextfloatsep{10.97\p@ \@plus 1.10\p@ \@minus 2.19\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 0.55fil}%
    \setlength\@fpsep{4.72\p@ \@plus 1.10fil}%
    \setlength\@fpbot{0.00\p@ \@plus 0.55fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 0.55fil}%
    \setlength\@dblfpsep{4.72\p@ \@plus 1.10fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 0.55fil}%
    %
    \setlength\partopsep{1.45\p@ \@plus 0.72\p@ \@minus 0.72\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 2.45\p@ \@plus 1.18\p@ \@minus 0.55\p@%
        \topsep 4.90\p@ \@plus 1.61\p@ \@minus 2.71\p@%
        \itemsep2.45\p@ \@plus 1.18\p@ \@minus 0.55\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 2.45\p@ \@plus 1.18\p@ \@minus 0.55\p@%
        \parsep 1.18\p@ \@plus 0.55\p@ \@minus 0.55\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 1.18\p@ \@plus 0.55\p@\@minus 0.55\p@%
        \parsep 0.00\p@%
        \partopsep 0.55\p@ \@plus 0.00\p@ \@minus 0.55\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 7 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeVII}{%
    \renewcommand\tiny{\@setfontsize\tiny{3.61}{4.25}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{4.89}{5.73}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{5.72}{6.87}%
        \abovedisplayskip 5.04\p@ \@plus 1.28\p@ \@minus 2.75\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 1.03\p@%
        \belowdisplayshortskip 2.72\p@ \@plus 1.24\p@ \@minus 1.47\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 2.72\p@ \@plus 1.05\p@ \@minus 1.05\p@%
            \parsep 1.47\p@ \@plus 0.83\p@ \@minus 0.64\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{6.35}{7.74}%
        \abovedisplayskip 6.24\p@ \@plus 1.92\p@ \@minus 3.16\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 1.69\p@%
        \belowdisplayshortskip 3.47\p@ \@plus 1.78\p@ \@minus 1.69\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 3.96\p@ \@plus 1.47\p@ \@minus 1.86\p@%
            \parsep 1.98\p@ \@plus 1.05\p@ \@minus 0.64\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{6.99}{8.48}%
        \abovedisplayskip 7.00\p@ \@plus 1.69\p@ \@minus 3.80\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 1.92\p@%
        \belowdisplayshortskip 4.04\p@ \@plus 2.12\p@ \@minus 1.92\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{8.15}{9.91}}%
    \renewcommand\Large{\@setfontsize\Large{9.72}{12.30}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{11.46}{14.66}}%
    \renewcommand\huge{\@setfontsize\huge{13.77}{16.97}}%
    \renewcommand\Huge{\@setfontsize\Huge{16.00}{19.20}}%
    %
    \setlength\smallskipamount{1.92\p@ \@plus 3.95\p@ \@minus 6.81\p@}%
    \setlength\medskipamount{3.84\p@ \@plus 1.28\p@ \@minus 1.28\p@}%
    \setlength\bigskipamount{7.68\p@ \@plus 2.56\p@ \@minus 2.56\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{7.68\p@}%
        \setlength\headsep{0.17in}%
        \setlength\topskip{7.00\p@}%
        \setlength\footskip{0.24in}%
        \setlength\@tempdimb{232.70\p@}%
    \fi%
    %
    \setlength\marginparsep{16.19\p@}%
    \setlength\marginparpush{3.59\p@}%
    %
    \setlength\footnotesep{4.82\p@}%
    \setlength{\skip\footins}{6.32\p@ \@plus 2.56\p@ \@minus 1.28\p@}%
    %
    \setlength\floatsep{7.68\p@ \@plus 1.28\p@ \@minus 1.67\p@}%
    \setlength\textfloatsep{12.80\p@ \@plus 1.28\p@ \@minus 2.56\p@}%
    \setlength\intextsep{8.07\p@ \@plus 1.67\p@ \@minus 1.67\p@}%
    %
    \setlength\dblfloatsep{8.07\p@ \@plus 1.28\p@ \@minus 1.67\p@}%
    \setlength\dbltextfloatsep{12.80\p@ \@plus 1.28\p@ \@minus 2.56\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 0.64fil}%
    \setlength\@fpsep{5.51\p@ \@plus 1.28fil}%
    \setlength\@fpbot{0.00\p@ \@plus 0.64fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 0.64fil}%
    \setlength\@dblfpsep{5.51\p@ \@plus 1.28fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 0.64fil}%
    %
    \setlength\partopsep{1.69\p@ \@plus 0.83\p@ \@minus 0.83\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 2.86\p@ \@plus 1.38\p@ \@minus 0.64\p@%
        \topsep 5.72\p@ \@plus 1.88\p@ \@minus 3.16\p@%
        \itemsep2.86\p@ \@plus 1.38\p@ \@minus 0.64\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 2.86\p@ \@plus 1.38\p@ \@minus 0.64\p@%
        \parsep 1.38\p@ \@plus 0.64\p@ \@minus 0.64\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 1.38\p@ \@plus 0.64\p@\@minus 0.64\p@%
        \parsep 0.00\p@%
        \partopsep 0.64\p@ \@plus 0.00\p@ \@minus 0.64\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 8 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeVIII}{%
    \renewcommand\tiny{\@setfontsize\tiny{4.12}{4.85}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{5.58}{6.55}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{6.54}{7.85}%
        \abovedisplayskip 5.76\p@ \@plus 1.46\p@ \@minus 3.15\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 1.18\p@%
        \belowdisplayshortskip 3.10\p@ \@plus 1.42\p@ \@minus 1.68\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 3.10\p@ \@plus 1.20\p@ \@minus 1.20\p@%
            \parsep 1.68\p@ \@plus 0.95\p@ \@minus 0.73\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{7.26}{8.85}%
        \abovedisplayskip 7.14\p@ \@plus 2.19\p@ \@minus 3.61\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 1.93\p@%
        \belowdisplayshortskip 3.97\p@ \@plus 2.04\p@ \@minus 1.93\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 4.52\p@ \@plus 1.68\p@ \@minus 2.13\p@%
            \parsep 2.26\p@ \@plus 1.20\p@ \@minus 0.73\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{7.99}{9.70}%
        \abovedisplayskip 8.00\p@ \@plus 1.93\p@ \@minus 4.34\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 2.19\p@%
        \belowdisplayshortskip 4.62\p@ \@plus 2.43\p@ \@minus 2.19\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{9.31}{11.33}}%
    \renewcommand\Large{\@setfontsize\Large{11.11}{14.05}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{13.10}{16.76}}%
    \renewcommand\huge{\@setfontsize\huge{15.74}{19.39}}%
    \renewcommand\Huge{\@setfontsize\Huge{18.28}{21.94}}%
    %
    \setlength\smallskipamount{2.19\p@ \@plus 3.46\p@ \@minus 5.96\p@}%
    \setlength\medskipamount{4.39\p@ \@plus 1.46\p@ \@minus 1.46\p@}%
    \setlength\bigskipamount{8.78\p@ \@plus 2.93\p@ \@minus 2.93\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{8.78\p@}%
        \setlength\headsep{0.19in}%
        \setlength\topskip{8.00\p@}%
        \setlength\footskip{0.28in}%
        \setlength\@tempdimb{265.94\p@}%
    \fi%
    %
    \setlength\marginparsep{14.17\p@}%
    \setlength\marginparpush{4.10\p@}%
    %
    \setlength\footnotesep{5.51\p@}%
    \setlength{\skip\footins}{7.22\p@ \@plus 2.93\p@ \@minus 1.46\p@}%
    %
    \setlength\floatsep{8.78\p@ \@plus 1.46\p@ \@minus 1.91\p@}%
    \setlength\textfloatsep{14.63\p@ \@plus 1.46\p@ \@minus 2.93\p@}%
    \setlength\intextsep{9.22\p@ \@plus 1.91\p@ \@minus 1.91\p@}%
    %
    \setlength\dblfloatsep{9.22\p@ \@plus 1.46\p@ \@minus 1.91\p@}%
    \setlength\dbltextfloatsep{14.63\p@ \@plus 1.46\p@ \@minus 2.93\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 0.73fil}%
    \setlength\@fpsep{6.29\p@ \@plus 1.46fil}%
    \setlength\@fpbot{0.00\p@ \@plus 0.73fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 0.73fil}%
    \setlength\@dblfpsep{6.29\p@ \@plus 1.46fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 0.73fil}%
    %
    \setlength\partopsep{1.93\p@ \@plus 0.95\p@ \@minus 0.95\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 3.27\p@ \@plus 1.57\p@ \@minus 0.73\p@%
        \topsep 6.54\p@ \@plus 2.15\p@ \@minus 3.61\p@%
        \itemsep3.27\p@ \@plus 1.57\p@ \@minus 0.73\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 3.27\p@ \@plus 1.57\p@ \@minus 0.73\p@%
        \parsep 1.57\p@ \@plus 0.73\p@ \@minus 0.73\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 1.57\p@ \@plus 0.73\p@\@minus 0.73\p@%
        \parsep 0.00\p@%
        \partopsep 0.73\p@ \@plus 0.00\p@ \@minus 0.73\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 9 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeIX}{%
    \renewcommand\tiny{\@setfontsize\tiny{4.64}{5.46}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{6.28}{7.37}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{7.35}{8.84}%
        \abovedisplayskip 6.48\p@ \@plus 1.65\p@ \@minus 3.54\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 1.32\p@%
        \belowdisplayshortskip 3.49\p@ \@plus 1.60\p@ \@minus 1.90\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 3.49\p@ \@plus 1.35\p@ \@minus 1.35\p@%
            \parsep 1.90\p@ \@plus 1.07\p@ \@minus 0.82\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{8.16}{9.96}%
        \abovedisplayskip 8.03\p@ \@plus 2.47\p@ \@minus 4.06\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 2.17\p@%
        \belowdisplayshortskip 4.46\p@ \@plus 2.29\p@ \@minus 2.17\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 5.09\p@ \@plus 1.90\p@ \@minus 2.40\p@%
            \parsep 2.54\p@ \@plus 1.35\p@ \@minus 0.82\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{8.99}{10.91}%
        \abovedisplayskip 9.00\p@ \@plus 2.17\p@ \@minus 4.89\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 2.47\p@%
        \belowdisplayshortskip 5.20\p@ \@plus 2.73\p@ \@minus 2.47\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{10.47}{12.75}}%
    \renewcommand\Large{\@setfontsize\Large{12.50}{15.81}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{14.74}{18.85}}%
    \renewcommand\huge{\@setfontsize\huge{17.70}{21.82}}%
    \renewcommand\Huge{\@setfontsize\Huge{20.57}{24.68}}%
    %
    \setlength\smallskipamount{2.47\p@ \@plus 3.07\p@ \@minus 5.30\p@}%
    \setlength\medskipamount{4.94\p@ \@plus 1.65\p@ \@minus 1.65\p@}%
    \setlength\bigskipamount{9.87\p@ \@plus 3.29\p@ \@minus 3.29\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{9.87\p@}%
        \setlength\headsep{0.22in}%
        \setlength\topskip{9.00\p@}%
        \setlength\footskip{0.31in}%
        \setlength\@tempdimb{299.18\p@}%
    \fi%
    %
    \setlength\marginparsep{12.59\p@}%
    \setlength\marginparpush{4.61\p@}%
    %
    \setlength\footnotesep{6.20\p@}%
    \setlength{\skip\footins}{8.13\p@ \@plus 3.29\p@ \@minus 1.65\p@}%
    %
    \setlength\floatsep{9.87\p@ \@plus 1.65\p@ \@minus 2.15\p@}%
    \setlength\textfloatsep{16.45\p@ \@plus 1.65\p@ \@minus 3.29\p@}%
    \setlength\intextsep{10.37\p@ \@plus 2.15\p@ \@minus 2.15\p@}%
    %
    \setlength\dblfloatsep{10.37\p@ \@plus 1.65\p@ \@minus 2.15\p@}%
    \setlength\dbltextfloatsep{16.45\p@ \@plus 1.65\p@ \@minus 3.29\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 0.82fil}%
    \setlength\@fpsep{7.08\p@ \@plus 1.65fil}%
    \setlength\@fpbot{0.00\p@ \@plus 0.82fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 0.82fil}%
    \setlength\@dblfpsep{7.08\p@ \@plus 1.65fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 0.82fil}%
    %
    \setlength\partopsep{2.17\p@ \@plus 1.07\p@ \@minus 1.07\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 3.68\p@ \@plus 1.77\p@ \@minus 0.82\p@%
        \topsep 7.35\p@ \@plus 2.42\p@ \@minus 4.06\p@%
        \itemsep3.68\p@ \@plus 1.77\p@ \@minus 0.82\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 3.68\p@ \@plus 1.77\p@ \@minus 0.82\p@%
        \parsep 1.77\p@ \@plus 0.82\p@ \@minus 0.82\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 1.77\p@ \@plus 0.82\p@\@minus 0.82\p@%
        \parsep 0.00\p@%
        \partopsep 0.82\p@ \@plus 0.00\p@ \@minus 0.82\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 10 (from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeX}{%
    \renewcommand\@ptsize{0}% last digit (for historical reasons)
    \renewcommand\tiny{\@setfontsize\tiny{5}{6}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{7}{8}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{8}{9.5}%
        \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@%
        \abovedisplayshortskip \z@ \@plus\p@%
        \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 3\p@ \@plus\p@ \@minus\p@%
            \parsep 2\p@ \@plus\p@ \@minus\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{9}{10.95}%
        \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@%
        \abovedisplayshortskip \z@ \@plus2\p@%
        \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 4\p@ \@plus2\p@ \@minus2\p@%
            \parsep 2\p@ \@plus\p@ \@minus\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{10}{12}%
        \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@%
        \abovedisplayshortskip \z@ \@plus3\p@%
        \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{12}{14.4}}%
    \renewcommand\Large{\@setfontsize\Large{14.4}{18}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{17}{22}}%
    \renewcommand\huge{\@setfontsize\huge{20}{25}}%
    \renewcommand\Huge{\@setfontsize\Huge{25}{30}}%
    %
    \setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}%
    \setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}%
    \setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{12\p@}%
        \setlength\headsep{.25in}%
        \setlength\topskip{10\p@}%
        \setlength\footskip{.35in}%
        \setlength\@tempdimb{345\p@}%
    \fi%
    %
    \setlength\marginparsep{11\p@}%
    \setlength\marginparpush{5\p@}%
    %
    \setlength\footnotesep{6.65\p@}%
    \setlength{\skip\footins}{9\p@ \@plus 4\p@ \@minus 2\p@}%
    %
    \setlength\floatsep{12\p@ \@plus 2\p@ \@minus 2\p@}%
    \setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}%
    \setlength\intextsep{12\p@ \@plus 2\p@ \@minus 2\p@}%
    %
    \setlength\dblfloatsep{12\p@ \@plus 2\p@ \@minus 2\p@}%
    \setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}%
    %
    \setlength\@fptop{0\p@ \@plus 1fil}%
    \setlength\@fpsep{8\p@ \@plus 2fil}%
    \setlength\@fpbot{0\p@ \@plus 1fil}%
    %
    \setlength\@dblfptop{0\p@ \@plus 1fil}%
    \setlength\@dblfpsep{8\p@ \@plus 2fil}%
    \setlength\@dblfpbot{0\p@ \@plus 1fil}%
    %
    \setlength\partopsep{2\p@ \@plus 1\p@ \@minus 1\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 4\p@ \@plus2\p@ \@minus\p@%
        \topsep 8\p@ \@plus2\p@ \@minus4\p@%
        \itemsep4\p@ \@plus2\p@ \@minus\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 4\p@ \@plus2\p@ \@minus\p@%
        \parsep 2\p@ \@plus\p@ \@minus\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 2\p@ \@plus\p@\@minus\p@%
        \parsep \z@%
        \partopsep \p@ \@plus\z@ \@minus\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
% Font size 11 (from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXI}{%
    \renewcommand\@ptsize{1}% last digit (for historical reasons)
    \renewcommand\tiny{\@setfontsize\tiny{6}{7}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{8}{9.5}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{9}{10.95}%
        \abovedisplayskip 8\p@ \@plus2\p@ \@minus4\p@%
        \abovedisplayshortskip \z@ \@plus\p@%
        \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{\leftmargin\leftmargini%
            \topsep 4\p@ \@plus2\p@ \@minus2\p@%
            \parsep 2\p@ \@plus\p@ \@minus\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{10}{12}%
        \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@%
        \abovedisplayshortskip \z@ \@plus3\p@%
        \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{\leftmargin\leftmargini%
            \topsep 6\p@ \@plus2\p@ \@minus2\p@%
            \parsep 3\p@ \@plus2\p@ \@minus\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{10.95}{13.6}%
        \abovedisplayskip 11\p@ \@plus3\p@ \@minus6\p@%
        \abovedisplayshortskip \z@ \@plus3\p@%
        \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{12}{14.4}}%
    \renewcommand\Large{\@setfontsize\Large{14.4}{18}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{17}{22}}%
    \renewcommand\huge{\@setfontsize\huge{20}{25}}%
    \renewcommand\Huge{\@setfontsize\Huge{25}{30}}%
    %
    \setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}%
    \setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}%
    \setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{12\p@}%
        \setlength\headsep{.275in}%
        \setlength\topskip{11\p@}%
        \setlength\footskip{.38in}%
        \setlength\@tempdimb{360\p@}%
    \fi%
    %
    \setlength\marginparsep{10\p@}%
    \setlength\marginparpush{5\p@}%
    %
    \setlength\footnotesep{7.7\p@}%
    \setlength{\skip\footins}{10\p@ \@plus 4\p@ \@minus 2\p@}%
    %
    \setlength\floatsep{12\p@ \@plus 2\p@ \@minus 2\p@}%
    \setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}%
    \setlength\intextsep{12\p@ \@plus 2\p@ \@minus 2\p@}%
    %
    \setlength\dblfloatsep{12\p@ \@plus 2\p@ \@minus 2\p@}%
    \setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}%
    %
    \setlength\@fptop{0\p@ \@plus 1fil}%
    \setlength\@fpsep{8\p@ \@plus 2fil}%
    \setlength\@fpbot{0\p@ \@plus 1fil}%
    %
    \setlength\@dblfptop{0\p@ \@plus 1fil}%
    \setlength\@dblfpsep{8\p@ \@plus 2fil}%
    \setlength\@dblfpbot{0\p@ \@plus 1fil}%
    %
    \setlength\partopsep{3\p@ \@plus 1\p@ \@minus 1\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 4.5\p@ \@plus2\p@ \@minus\p@%
        \topsep 9\p@   \@plus3\p@ \@minus5\p@%
        \itemsep4.5\p@ \@plus2\p@ \@minus\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 4.5\p@ \@plus2\p@ \@minus\p@%
        \parsep 2\p@ \@plus\p@ \@minus\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 2\p@ \@plus\p@\@minus\p@%
        \parsep \z@%
        \partopsep \p@ \@plus\z@ \@minus\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}
% Font size 12 (from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXII}{%
    \renewcommand\@ptsize{2}% last digit (for historical reasons)
    \renewcommand\tiny{\@setfontsize\tiny{6}{7}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{8}{9.5}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{10}{12}%
        \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@%
        \abovedisplayshortskip \z@ \@plus3\p@%
        \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{\leftmargin\leftmargini%
            \topsep 6\p@ \@plus2\p@ \@minus2\p@%
            \parsep 3\p@ \@plus2\p@ \@minus\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{10.95}{13.6}%
        \abovedisplayskip 11\p@ \@plus3\p@ \@minus6\p@%
        \abovedisplayshortskip \z@ \@plus3\p@%
        \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{\leftmargin\leftmargini%
            \topsep 9\p@ \@plus3\p@ \@minus5\p@%
            \parsep 4.5\p@ \@plus2\p@ \@minus\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{12}{14.4}%
        \abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@%
        \abovedisplayshortskip \z@ \@plus3\p@%
        \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{14.4}{18}}%
    \renewcommand\Large{\@setfontsize\Large{17}{22}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{20}{25}}%
    \renewcommand\huge{\@setfontsize\huge{25}{30}}%
    \renewcommand\Huge{\@setfontsize\Huge{25}{30}}%
    %
    \setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}%
    \setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}%
    \setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{12\p@}%
        \setlength\headsep{.275in}%
        \setlength\topskip{12\p@}%
        \setlength\footskip{30\p@}%
        \setlength\@tempdimb{390\p@}%
    \fi%
    %
    \setlength\marginparsep{10\p@}%
    \setlength\marginparpush{7\p@}%
    %
    \setlength\footnotesep{8.4\p@}%
    \setlength{\skip\footins}{10.8\p@ \@plus 4\p@ \@minus 2\p@}%
    %
    \setlength\floatsep{12\p@ \@plus 2\p@ \@minus 4\p@}%
    \setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}%
    \setlength\intextsep{14\p@ \@plus 4\p@ \@minus 4\p@}%
    %
    \setlength\dblfloatsep{14\p@ \@plus 2\p@ \@minus 4\p@}%
    \setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}%
    %
    \setlength\@fptop{0\p@ \@plus 1fil}%
    \setlength\@fpsep{10\p@ \@plus 2fil}%
    \setlength\@fpbot{0\p@ \@plus 1fil}%
    %
    \setlength\@dblfptop{0\p@ \@plus 1fil}%
    \setlength\@dblfpsep{10\p@ \@plus 2fil}%
    \setlength\@dblfpbot{0\p@ \@plus 1fil}%
    %
    \setlength\partopsep{3\p@ \@plus 2\p@ \@minus 2\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 5\p@  \@plus2.5\p@ \@minus\p@%
        \topsep 10\p@ \@plus4\p@   \@minus6\p@%
        \itemsep5\p@  \@plus2.5\p@ \@minus\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 5\p@ \@plus2.5\p@ \@minus\p@%
        \parsep 2.5\p@ \@plus\p@ \@minus\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 2.5\p@ \@plus\p@\@minus\p@%
        \parsep \z@%
        \partopsep \p@ \@plus\z@ \@minus\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 13 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXIII}{%
    \renewcommand\tiny{\@setfontsize\tiny{6.70}{7.89}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{9.07}{10.64}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{10.62}{12.76}%
        \abovedisplayskip 9.36\p@ \@plus 2.38\p@ \@minus 5.11\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 1.91\p@%
        \belowdisplayshortskip 5.04\p@ \@plus 2.30\p@ \@minus 2.74\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 5.04\p@ \@plus 1.94\p@ \@minus 1.94\p@%
            \parsep 2.74\p@ \@plus 1.55\p@ \@minus 1.19\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{11.79}{14.38}%
        \abovedisplayskip 11.59\p@ \@plus 3.57\p@ \@minus 5.87\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 3.13\p@%
        \belowdisplayshortskip 6.44\p@ \@plus 3.31\p@ \@minus 3.13\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 7.35\p@ \@plus 2.74\p@ \@minus 3.46\p@%
            \parsep 3.67\p@ \@plus 1.94\p@ \@minus 1.19\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{12.98}{15.76}%
        \abovedisplayskip 13.00\p@ \@plus 3.13\p@ \@minus 7.06\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 3.57\p@%
        \belowdisplayshortskip 7.51\p@ \@plus 3.94\p@ \@minus 3.57\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{15.13}{18.41}}%
    \renewcommand\Large{\@setfontsize\Large{18.05}{22.84}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{21.29}{27.23}}%
    \renewcommand\huge{\@setfontsize\huge{25.57}{31.52}}%
    \renewcommand\Huge{\@setfontsize\Huge{29.71}{35.65}}%
    %
    \setlength\smallskipamount{3.57\p@ \@plus 2.13\p@ \@minus 3.67\p@}%
    \setlength\medskipamount{7.13\p@ \@plus 2.38\p@ \@minus 2.38\p@}%
    \setlength\bigskipamount{14.26\p@ \@plus 4.75\p@ \@minus 4.75\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{14.26\p@}%
        \setlength\headsep{0.32in}%
        \setlength\topskip{13.00\p@}%
        \setlength\footskip{0.45in}%
        \setlength\@tempdimb{432.15\p@}%
    \fi%
    %
    \setlength\marginparsep{8.72\p@}%
    \setlength\marginparpush{6.66\p@}%
    %
    \setlength\footnotesep{8.95\p@}%
    \setlength{\skip\footins}{11.74\p@ \@plus 4.75\p@ \@minus 2.38\p@}%
    %
    \setlength\floatsep{14.26\p@ \@plus 2.38\p@ \@minus 3.10\p@}%
    \setlength\textfloatsep{23.77\p@ \@plus 2.38\p@ \@minus 4.75\p@}%
    \setlength\intextsep{14.98\p@ \@plus 3.10\p@ \@minus 3.10\p@}%
    %
    \setlength\dblfloatsep{14.98\p@ \@plus 2.38\p@ \@minus 3.10\p@}%
    \setlength\dbltextfloatsep{23.77\p@ \@plus 2.38\p@ \@minus 4.75\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 1.19fil}%
    \setlength\@fpsep{10.23\p@ \@plus 2.38fil}%
    \setlength\@fpbot{0.00\p@ \@plus 1.19fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 1.19fil}%
    \setlength\@dblfpsep{10.23\p@ \@plus 2.38fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 1.19fil}%
    %
    \setlength\partopsep{3.13\p@ \@plus 1.55\p@ \@minus 1.55\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 5.31\p@ \@plus 2.56\p@ \@minus 1.19\p@%
        \topsep 10.62\p@ \@plus 3.49\p@ \@minus 5.87\p@%
        \itemsep5.31\p@ \@plus 2.56\p@ \@minus 1.19\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 5.31\p@ \@plus 2.56\p@ \@minus 1.19\p@%
        \parsep 2.56\p@ \@plus 1.19\p@ \@minus 1.19\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 2.56\p@ \@plus 1.19\p@\@minus 1.19\p@%
        \parsep 0.00\p@%
        \partopsep 1.19\p@ \@plus 0.00\p@ \@minus 1.19\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 14 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXIV}{%
    \renewcommand\tiny{\@setfontsize\tiny{7.21}{8.49}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{9.77}{11.46}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{11.44}{13.75}%
        \abovedisplayskip 10.08\p@ \@plus 2.56\p@ \@minus 5.51\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 2.06\p@%
        \belowdisplayshortskip 5.43\p@ \@plus 2.48\p@ \@minus 2.95\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 5.43\p@ \@plus 2.09\p@ \@minus 2.09\p@%
            \parsep 2.95\p@ \@plus 1.67\p@ \@minus 1.28\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{12.70}{15.49}%
        \abovedisplayskip 12.49\p@ \@plus 3.84\p@ \@minus 6.32\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 3.37\p@%
        \belowdisplayshortskip 6.94\p@ \@plus 3.57\p@ \@minus 3.37\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 7.91\p@ \@plus 2.95\p@ \@minus 3.73\p@%
            \parsep 3.96\p@ \@plus 2.09\p@ \@minus 1.28\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{13.98}{16.97}%
        \abovedisplayskip 14.00\p@ \@plus 3.37\p@ \@minus 7.60\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 3.84\p@%
        \belowdisplayshortskip 8.09\p@ \@plus 4.25\p@ \@minus 3.84\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{16.29}{19.83}}%
    \renewcommand\Large{\@setfontsize\Large{19.44}{24.59}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{22.92}{29.32}}%
    \renewcommand\huge{\@setfontsize\huge{27.54}{33.94}}%
    \renewcommand\Huge{\@setfontsize\Huge{31.99}{38.39}}%
    %
    \setlength\smallskipamount{3.84\p@ \@plus 1.98\p@ \@minus 3.40\p@}%
    \setlength\medskipamount{7.68\p@ \@plus 2.56\p@ \@minus 2.56\p@}%
    \setlength\bigskipamount{15.36\p@ \@plus 5.12\p@ \@minus 5.12\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{15.36\p@}%
        \setlength\headsep{0.34in}%
        \setlength\topskip{14.00\p@}%
        \setlength\footskip{0.49in}%
        \setlength\@tempdimb{465.39\p@}%
    \fi%
    %
    \setlength\marginparsep{8.10\p@}%
    \setlength\marginparpush{7.18\p@}%
    %
    \setlength\footnotesep{9.64\p@}%
    \setlength{\skip\footins}{12.64\p@ \@plus 5.12\p@ \@minus 2.56\p@}%
    %
    \setlength\floatsep{15.36\p@ \@plus 2.56\p@ \@minus 3.34\p@}%
    \setlength\textfloatsep{25.60\p@ \@plus 2.56\p@ \@minus 5.12\p@}%
    \setlength\intextsep{16.14\p@ \@plus 3.34\p@ \@minus 3.34\p@}%
    %
    \setlength\dblfloatsep{16.14\p@ \@plus 2.56\p@ \@minus 3.34\p@}%
    \setlength\dbltextfloatsep{25.60\p@ \@plus 2.56\p@ \@minus 5.12\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 1.28fil}%
    \setlength\@fpsep{11.02\p@ \@plus 2.56fil}%
    \setlength\@fpbot{0.00\p@ \@plus 1.28fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 1.28fil}%
    \setlength\@dblfpsep{11.02\p@ \@plus 2.56fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 1.28fil}%
    %
    \setlength\partopsep{3.37\p@ \@plus 1.67\p@ \@minus 1.67\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 5.72\p@ \@plus 2.75\p@ \@minus 1.28\p@%
        \topsep 11.44\p@ \@plus 3.76\p@ \@minus 6.32\p@%
        \itemsep5.72\p@ \@plus 2.75\p@ \@minus 1.28\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 5.72\p@ \@plus 2.75\p@ \@minus 1.28\p@%
        \parsep 2.75\p@ \@plus 1.28\p@ \@minus 1.28\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 2.75\p@ \@plus 1.28\p@\@minus 1.28\p@%
        \parsep 0.00\p@%
        \partopsep 1.28\p@ \@plus 0.00\p@ \@minus 1.28\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 15 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXV}{%
    \renewcommand\tiny{\@setfontsize\tiny{7.73}{9.10}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{10.47}{12.28}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{12.26}{14.73}%
        \abovedisplayskip 10.80\p@ \@plus 2.74\p@ \@minus 5.90\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 2.20\p@%
        \belowdisplayshortskip 5.82\p@ \@plus 2.66\p@ \@minus 3.16\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 5.82\p@ \@plus 2.24\p@ \@minus 2.24\p@%
            \parsep 3.16\p@ \@plus 1.79\p@ \@minus 1.37\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{13.61}{16.60}%
        \abovedisplayskip 13.38\p@ \@plus 4.11\p@ \@minus 6.77\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 3.61\p@%
        \belowdisplayshortskip 7.44\p@ \@plus 3.82\p@ \@minus 3.61\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 8.48\p@ \@plus 3.16\p@ \@minus 3.99\p@%
            \parsep 4.24\p@ \@plus 2.24\p@ \@minus 1.37\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{14.98}{18.18}%
        \abovedisplayskip 15.00\p@ \@plus 3.61\p@ \@minus 8.14\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 4.11\p@%
        \belowdisplayshortskip 8.66\p@ \@plus 4.55\p@ \@minus 4.11\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{17.45}{21.25}}%
    \renewcommand\Large{\@setfontsize\Large{20.83}{26.35}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{24.56}{31.42}}%
    \renewcommand\huge{\@setfontsize\huge{29.51}{36.36}}%
    \renewcommand\Huge{\@setfontsize\Huge{34.28}{41.14}}%
    %
    \setlength\smallskipamount{4.11\p@ \@plus 1.84\p@ \@minus 3.18\p@}%
    \setlength\medskipamount{8.23\p@ \@plus 2.74\p@ \@minus 2.74\p@}%
    \setlength\bigskipamount{16.45\p@ \@plus 5.48\p@ \@minus 5.48\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{16.45\p@}%
        \setlength\headsep{0.36in}%
        \setlength\topskip{15.00\p@}%
        \setlength\footskip{0.52in}%
        \setlength\@tempdimb{498.64\p@}%
    \fi%
    %
    \setlength\marginparsep{7.56\p@}%
    \setlength\marginparpush{7.69\p@}%
    %
    \setlength\footnotesep{10.32\p@}%
    \setlength{\skip\footins}{13.55\p@ \@plus 5.48\p@ \@minus 2.74\p@}%
    %
    \setlength\floatsep{16.45\p@ \@plus 2.74\p@ \@minus 3.58\p@}%
    \setlength\textfloatsep{27.42\p@ \@plus 2.74\p@ \@minus 5.48\p@}%
    \setlength\intextsep{17.29\p@ \@plus 3.58\p@ \@minus 3.58\p@}%
    %
    \setlength\dblfloatsep{17.29\p@ \@plus 2.74\p@ \@minus 3.58\p@}%
    \setlength\dbltextfloatsep{27.42\p@ \@plus 2.74\p@ \@minus 5.48\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 1.37fil}%
    \setlength\@fpsep{11.80\p@ \@plus 2.74fil}%
    \setlength\@fpbot{0.00\p@ \@plus 1.37fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 1.37fil}%
    \setlength\@dblfpsep{11.80\p@ \@plus 2.74fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 1.37fil}%
    %
    \setlength\partopsep{3.61\p@ \@plus 1.79\p@ \@minus 1.79\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 6.13\p@ \@plus 2.95\p@ \@minus 1.37\p@%
        \topsep 12.26\p@ \@plus 4.03\p@ \@minus 6.77\p@%
        \itemsep6.13\p@ \@plus 2.95\p@ \@minus 1.37\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 6.13\p@ \@plus 2.95\p@ \@minus 1.37\p@%
        \parsep 2.95\p@ \@plus 1.37\p@ \@minus 1.37\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 2.95\p@ \@plus 1.37\p@\@minus 1.37\p@%
        \parsep 0.00\p@%
        \partopsep 1.37\p@ \@plus 0.00\p@ \@minus 1.37\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 16 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXVI}{%
    \renewcommand\tiny{\@setfontsize\tiny{8.24}{9.71}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{11.17}{13.09}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{13.07}{15.71}%
        \abovedisplayskip 11.52\p@ \@plus 2.93\p@ \@minus 6.29\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 2.35\p@%
        \belowdisplayshortskip 6.21\p@ \@plus 2.84\p@ \@minus 3.37\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 6.21\p@ \@plus 2.39\p@ \@minus 2.39\p@%
            \parsep 3.37\p@ \@plus 1.91\p@ \@minus 1.46\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{14.52}{17.70}%
        \abovedisplayskip 14.27\p@ \@plus 4.39\p@ \@minus 7.22\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 3.85\p@%
        \belowdisplayshortskip 7.93\p@ \@plus 4.08\p@ \@minus 3.85\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 9.04\p@ \@plus 3.37\p@ \@minus 4.26\p@%
            \parsep 4.52\p@ \@plus 2.39\p@ \@minus 1.46\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{15.98}{19.39}%
        \abovedisplayskip 16.00\p@ \@plus 3.85\p@ \@minus 8.69\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 4.39\p@%
        \belowdisplayshortskip 9.24\p@ \@plus 4.85\p@ \@minus 4.39\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{18.62}{22.66}}%
    \renewcommand\Large{\@setfontsize\Large{22.22}{28.11}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{26.20}{33.51}}%
    \renewcommand\huge{\@setfontsize\huge{31.47}{38.79}}%
    \renewcommand\Huge{\@setfontsize\Huge{36.57}{43.88}}%
    %
    \setlength\smallskipamount{4.39\p@ \@plus 1.73\p@ \@minus 2.98\p@}%
    \setlength\medskipamount{8.78\p@ \@plus 2.93\p@ \@minus 2.93\p@}%
    \setlength\bigskipamount{17.55\p@ \@plus 5.85\p@ \@minus 5.85\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{17.55\p@}%
        \setlength\headsep{0.39in}%
        \setlength\topskip{16.00\p@}%
        \setlength\footskip{0.56in}%
        \setlength\@tempdimb{531.88\p@}%
    \fi%
    %
    \setlength\marginparsep{7.08\p@}%
    \setlength\marginparpush{8.20\p@}%
    %
    \setlength\footnotesep{11.01\p@}%
    \setlength{\skip\footins}{14.45\p@ \@plus 5.85\p@ \@minus 2.93\p@}%
    %
    \setlength\floatsep{17.55\p@ \@plus 2.93\p@ \@minus 3.81\p@}%
    \setlength\textfloatsep{29.25\p@ \@plus 2.93\p@ \@minus 5.85\p@}%
    \setlength\intextsep{18.44\p@ \@plus 3.81\p@ \@minus 3.81\p@}%
    %
    \setlength\dblfloatsep{18.44\p@ \@plus 2.93\p@ \@minus 3.81\p@}%
    \setlength\dbltextfloatsep{29.25\p@ \@plus 2.93\p@ \@minus 5.85\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 1.46fil}%
    \setlength\@fpsep{12.59\p@ \@plus 2.93fil}%
    \setlength\@fpbot{0.00\p@ \@plus 1.46fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 1.46fil}%
    \setlength\@dblfpsep{12.59\p@ \@plus 2.93fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 1.46fil}%
    %
    \setlength\partopsep{3.85\p@ \@plus 1.91\p@ \@minus 1.91\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 6.54\p@ \@plus 3.15\p@ \@minus 1.46\p@%
        \topsep 13.07\p@ \@plus 4.30\p@ \@minus 7.22\p@%
        \itemsep6.54\p@ \@plus 3.15\p@ \@minus 1.46\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 6.54\p@ \@plus 3.15\p@ \@minus 1.46\p@%
        \parsep 3.15\p@ \@plus 1.46\p@ \@minus 1.46\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 3.15\p@ \@plus 1.46\p@\@minus 1.46\p@%
        \parsep 0.00\p@%
        \partopsep 1.46\p@ \@plus 0.00\p@ \@minus 1.46\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 18 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXVIII}{%
    \renewcommand\tiny{\@setfontsize\tiny{9.27}{10.92}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{12.56}{14.73}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{14.71}{17.67}%
        \abovedisplayskip 12.96\p@ \@plus 3.29\p@ \@minus 7.08\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 2.65\p@%
        \belowdisplayshortskip 6.98\p@ \@plus 3.19\p@ \@minus 3.79\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 6.98\p@ \@plus 2.69\p@ \@minus 2.69\p@%
            \parsep 3.79\p@ \@plus 2.15\p@ \@minus 1.65\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{16.33}{19.92}%
        \abovedisplayskip 16.05\p@ \@plus 4.94\p@ \@minus 8.13\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 4.34\p@%
        \belowdisplayshortskip 8.92\p@ \@plus 4.59\p@ \@minus 4.34\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 10.17\p@ \@plus 3.79\p@ \@minus 4.79\p@%
            \parsep 5.09\p@ \@plus 2.69\p@ \@minus 1.65\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{17.97}{21.82}%
        \abovedisplayskip 18.00\p@ \@plus 4.34\p@ \@minus 9.77\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 4.94\p@%
        \belowdisplayshortskip 10.40\p@ \@plus 5.46\p@ \@minus 4.94\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{20.95}{25.49}}%
    \renewcommand\Large{\@setfontsize\Large{24.99}{31.62}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{29.47}{37.70}}%
    \renewcommand\huge{\@setfontsize\huge{35.41}{43.64}}%
    \renewcommand\Huge{\@setfontsize\Huge{41.14}{49.36}}%
    %
    \setlength\smallskipamount{4.94\p@ \@plus 1.54\p@ \@minus 2.65\p@}%
    \setlength\medskipamount{9.87\p@ \@plus 3.29\p@ \@minus 3.29\p@}%
    \setlength\bigskipamount{19.75\p@ \@plus 6.58\p@ \@minus 6.58\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{19.75\p@}%
        \setlength\headsep{0.44in}%
        \setlength\topskip{18.00\p@}%
        \setlength\footskip{0.62in}%
        \setlength\@tempdimb{598.36\p@}%
    \fi%
    %
    \setlength\marginparsep{6.30\p@}%
    \setlength\marginparpush{9.23\p@}%
    %
    \setlength\footnotesep{12.39\p@}%
    \setlength{\skip\footins}{16.25\p@ \@plus 6.58\p@ \@minus 3.29\p@}%
    %
    \setlength\floatsep{19.75\p@ \@plus 3.29\p@ \@minus 4.29\p@}%
    \setlength\textfloatsep{32.91\p@ \@plus 3.29\p@ \@minus 6.58\p@}%
    \setlength\intextsep{20.75\p@ \@plus 4.29\p@ \@minus 4.29\p@}%
    %
    \setlength\dblfloatsep{20.75\p@ \@plus 3.29\p@ \@minus 4.29\p@}%
    \setlength\dbltextfloatsep{32.91\p@ \@plus 3.29\p@ \@minus 6.58\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 1.65fil}%
    \setlength\@fpsep{14.16\p@ \@plus 3.29fil}%
    \setlength\@fpbot{0.00\p@ \@plus 1.65fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 1.65fil}%
    \setlength\@dblfpsep{14.16\p@ \@plus 3.29fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 1.65fil}%
    %
    \setlength\partopsep{4.34\p@ \@plus 2.15\p@ \@minus 2.15\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 7.35\p@ \@plus 3.54\p@ \@minus 1.65\p@%
        \topsep 14.71\p@ \@plus 4.84\p@ \@minus 8.13\p@%
        \itemsep7.35\p@ \@plus 3.54\p@ \@minus 1.65\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 7.35\p@ \@plus 3.54\p@ \@minus 1.65\p@%
        \parsep 3.54\p@ \@plus 1.65\p@ \@minus 1.65\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 3.54\p@ \@plus 1.65\p@\@minus 1.65\p@%
        \parsep 0.00\p@%
        \partopsep 1.65\p@ \@plus 0.00\p@ \@minus 1.65\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 20 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXX}{%
    \renewcommand\tiny{\@setfontsize\tiny{10.30}{12.13}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{13.96}{16.37}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{16.34}{19.64}%
        \abovedisplayskip 14.40\p@ \@plus 3.66\p@ \@minus 7.87\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 2.94\p@%
        \belowdisplayshortskip 7.76\p@ \@plus 3.55\p@ \@minus 4.21\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 7.76\p@ \@plus 2.99\p@ \@minus 2.99\p@%
            \parsep 4.21\p@ \@plus 2.38\p@ \@minus 1.83\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{18.14}{22.13}%
        \abovedisplayskip 17.84\p@ \@plus 5.48\p@ \@minus 9.03\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 4.82\p@%
        \belowdisplayshortskip 9.91\p@ \@plus 5.10\p@ \@minus 4.82\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 11.30\p@ \@plus 4.21\p@ \@minus 5.32\p@%
            \parsep 5.65\p@ \@plus 2.99\p@ \@minus 1.83\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{19.97}{24.24}%
        \abovedisplayskip 20.00\p@ \@plus 4.82\p@ \@minus 10.86\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 5.48\p@%
        \belowdisplayshortskip 11.55\p@ \@plus 6.07\p@ \@minus 5.48\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{23.27}{28.33}}%
    \renewcommand\Large{\@setfontsize\Large{27.77}{35.13}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{32.75}{41.89}}%
    \renewcommand\huge{\@setfontsize\huge{39.34}{48.48}}%
    \renewcommand\Huge{\@setfontsize\Huge{45.71}{54.85}}%
    %
    \setlength\smallskipamount{5.48\p@ \@plus 1.38\p@ \@minus 2.38\p@}%
    \setlength\medskipamount{10.97\p@ \@plus 3.66\p@ \@minus 3.66\p@}%
    \setlength\bigskipamount{21.94\p@ \@plus 7.31\p@ \@minus 7.31\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{21.94\p@}%
        \setlength\headsep{0.49in}%
        \setlength\topskip{20.00\p@}%
        \setlength\footskip{0.69in}%
        \setlength\@tempdimb{664.85\p@}%
    \fi%
    %
    \setlength\marginparsep{5.67\p@}%
    \setlength\marginparpush{10.25\p@}%
    %
    \setlength\footnotesep{13.77\p@}%
    \setlength{\skip\footins}{18.06\p@ \@plus 7.31\p@ \@minus 3.66\p@}%
    %
    \setlength\floatsep{21.94\p@ \@plus 3.66\p@ \@minus 4.77\p@}%
    \setlength\textfloatsep{36.57\p@ \@plus 3.66\p@ \@minus 7.31\p@}%
    \setlength\intextsep{23.05\p@ \@plus 4.77\p@ \@minus 4.77\p@}%
    %
    \setlength\dblfloatsep{23.05\p@ \@plus 3.66\p@ \@minus 4.77\p@}%
    \setlength\dbltextfloatsep{36.57\p@ \@plus 3.66\p@ \@minus 7.31\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 1.83fil}%
    \setlength\@fpsep{15.74\p@ \@plus 3.66fil}%
    \setlength\@fpbot{0.00\p@ \@plus 1.83fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 1.83fil}%
    \setlength\@dblfpsep{15.74\p@ \@plus 3.66fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 1.83fil}%
    %
    \setlength\partopsep{4.82\p@ \@plus 2.38\p@ \@minus 2.38\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 8.17\p@ \@plus 3.93\p@ \@minus 1.83\p@%
        \topsep 16.34\p@ \@plus 5.37\p@ \@minus 9.03\p@%
        \itemsep8.17\p@ \@plus 3.93\p@ \@minus 1.83\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 8.17\p@ \@plus 3.93\p@ \@minus 1.83\p@%
        \parsep 3.93\p@ \@plus 1.83\p@ \@minus 1.83\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 3.93\p@ \@plus 1.83\p@\@minus 1.83\p@%
        \parsep 0.00\p@%
        \partopsep 1.83\p@ \@plus 0.00\p@ \@minus 1.83\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 22 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXXII}{%
    \renewcommand\tiny{\@setfontsize\tiny{11.33}{13.34}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{15.36}{18.01}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{17.98}{21.60}%
        \abovedisplayskip 15.84\p@ \@plus 4.02\p@ \@minus 8.66\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 3.23\p@%
        \belowdisplayshortskip 8.53\p@ \@plus 3.90\p@ \@minus 4.63\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 8.53\p@ \@plus 3.29\p@ \@minus 3.29\p@%
            \parsep 4.63\p@ \@plus 2.62\p@ \@minus 2.01\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{19.96}{24.34}%
        \abovedisplayskip 19.62\p@ \@plus 6.03\p@ \@minus 9.93\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 5.30\p@%
        \belowdisplayshortskip 10.91\p@ \@plus 5.61\p@ \@minus 5.30\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 12.43\p@ \@plus 4.63\p@ \@minus 5.86\p@%
            \parsep 6.22\p@ \@plus 3.29\p@ \@minus 2.01\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{21.97}{26.67}%
        \abovedisplayskip 22.00\p@ \@plus 5.30\p@ \@minus 11.94\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 6.03\p@%
        \belowdisplayshortskip 12.71\p@ \@plus 6.67\p@ \@minus 6.03\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{25.60}{31.16}}%
    \renewcommand\Large{\@setfontsize\Large{30.55}{38.64}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{36.02}{46.08}}%
    \renewcommand\huge{\@setfontsize\huge{43.28}{53.33}}%
    \renewcommand\Huge{\@setfontsize\Huge{50.28}{60.33}}%
    %
    \setlength\smallskipamount{6.03\p@ \@plus 1.26\p@ \@minus 2.17\p@}%
    \setlength\medskipamount{12.07\p@ \@plus 4.02\p@ \@minus 4.02\p@}%
    \setlength\bigskipamount{24.13\p@ \@plus 8.04\p@ \@minus 8.04\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{24.13\p@}%
        \setlength\headsep{0.53in}%
        \setlength\topskip{22.00\p@}%
        \setlength\footskip{0.76in}%
        \setlength\@tempdimb{731.33\p@}%
    \fi%
    %
    \setlength\marginparsep{5.15\p@}%
    \setlength\marginparpush{11.28\p@}%
    %
    \setlength\footnotesep{15.14\p@}%
    \setlength{\skip\footins}{19.87\p@ \@plus 8.04\p@ \@minus 4.02\p@}%
    %
    \setlength\floatsep{24.13\p@ \@plus 4.02\p@ \@minus 5.24\p@}%
    \setlength\textfloatsep{40.22\p@ \@plus 4.02\p@ \@minus 8.04\p@}%
    \setlength\intextsep{25.36\p@ \@plus 5.24\p@ \@minus 5.24\p@}%
    %
    \setlength\dblfloatsep{25.36\p@ \@plus 4.02\p@ \@minus 5.24\p@}%
    \setlength\dbltextfloatsep{40.22\p@ \@plus 4.02\p@ \@minus 8.04\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 2.01fil}%
    \setlength\@fpsep{17.31\p@ \@plus 4.02fil}%
    \setlength\@fpbot{0.00\p@ \@plus 2.01fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 2.01fil}%
    \setlength\@dblfpsep{17.31\p@ \@plus 4.02fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 2.01fil}%
    %
    \setlength\partopsep{5.30\p@ \@plus 2.62\p@ \@minus 2.62\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 8.99\p@ \@plus 4.33\p@ \@minus 2.01\p@%
        \topsep 17.98\p@ \@plus 5.91\p@ \@minus 9.93\p@%
        \itemsep8.99\p@ \@plus 4.33\p@ \@minus 2.01\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 8.99\p@ \@plus 4.33\p@ \@minus 2.01\p@%
        \parsep 4.33\p@ \@plus 2.01\p@ \@minus 2.01\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 4.33\p@ \@plus 2.01\p@\@minus 2.01\p@%
        \parsep 0.00\p@%
        \partopsep 2.01\p@ \@plus 0.00\p@ \@minus 2.01\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 24 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXXIV}{%
    \renewcommand\tiny{\@setfontsize\tiny{12.36}{14.56}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{16.75}{19.64}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{19.61}{23.56}%
        \abovedisplayskip 17.28\p@ \@plus 4.39\p@ \@minus 9.44\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 3.53\p@%
        \belowdisplayshortskip 9.31\p@ \@plus 4.25\p@ \@minus 5.05\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 9.31\p@ \@plus 3.59\p@ \@minus 3.59\p@%
            \parsep 5.05\p@ \@plus 2.86\p@ \@minus 2.19\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{21.77}{26.55}%
        \abovedisplayskip 21.41\p@ \@plus 6.58\p@ \@minus 10.84\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 5.78\p@%
        \belowdisplayshortskip 11.90\p@ \@plus 6.12\p@ \@minus 5.78\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 13.56\p@ \@plus 5.05\p@ \@minus 6.39\p@%
            \parsep 6.78\p@ \@plus 3.59\p@ \@minus 2.19\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{23.96}{29.09}%
        \abovedisplayskip 24.00\p@ \@plus 5.78\p@ \@minus 13.03\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 6.58\p@%
        \belowdisplayshortskip 13.86\p@ \@plus 7.28\p@ \@minus 6.58\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{27.93}{33.99}}%
    \renewcommand\Large{\@setfontsize\Large{33.33}{42.16}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{39.30}{50.27}}%
    \renewcommand\huge{\@setfontsize\huge{47.21}{58.18}}%
    \renewcommand\Huge{\@setfontsize\Huge{54.85}{65.82}}%
    %
    \setlength\smallskipamount{6.58\p@ \@plus 1.15\p@ \@minus 1.99\p@}%
    \setlength\medskipamount{13.16\p@ \@plus 4.39\p@ \@minus 4.39\p@}%
    \setlength\bigskipamount{26.33\p@ \@plus 8.78\p@ \@minus 8.78\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{26.33\p@}%
        \setlength\headsep{0.58in}%
        \setlength\topskip{24.00\p@}%
        \setlength\footskip{0.83in}%
        \setlength\@tempdimb{797.82\p@}%
    \fi%
    %
    \setlength\marginparsep{4.72\p@}%
    \setlength\marginparpush{12.30\p@}%
    %
    \setlength\footnotesep{16.52\p@}%
    \setlength{\skip\footins}{21.67\p@ \@plus 8.78\p@ \@minus 4.39\p@}%
    %
    \setlength\floatsep{26.33\p@ \@plus 4.39\p@ \@minus 5.72\p@}%
    \setlength\textfloatsep{43.88\p@ \@plus 4.39\p@ \@minus 8.78\p@}%
    \setlength\intextsep{27.66\p@ \@plus 5.72\p@ \@minus 5.72\p@}%
    %
    \setlength\dblfloatsep{27.66\p@ \@plus 4.39\p@ \@minus 5.72\p@}%
    \setlength\dbltextfloatsep{43.88\p@ \@plus 4.39\p@ \@minus 8.78\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 2.19fil}%
    \setlength\@fpsep{18.88\p@ \@plus 4.39fil}%
    \setlength\@fpbot{0.00\p@ \@plus 2.19fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 2.19fil}%
    \setlength\@dblfpsep{18.88\p@ \@plus 4.39fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 2.19fil}%
    %
    \setlength\partopsep{5.78\p@ \@plus 2.86\p@ \@minus 2.86\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 9.81\p@ \@plus 4.72\p@ \@minus 2.19\p@%
        \topsep 19.61\p@ \@plus 6.45\p@ \@minus 10.84\p@%
        \itemsep9.81\p@ \@plus 4.72\p@ \@minus 2.19\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 9.81\p@ \@plus 4.72\p@ \@minus 2.19\p@%
        \parsep 4.72\p@ \@plus 2.19\p@ \@minus 2.19\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 4.72\p@ \@plus 2.19\p@\@minus 2.19\p@%
        \parsep 0.00\p@%
        \partopsep 2.19\p@ \@plus 0.00\p@ \@minus 2.19\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 28 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXXVIII}{%
    \renewcommand\tiny{\@setfontsize\tiny{14.42}{16.98}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{19.54}{22.92}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{22.88}{27.49}%
        \abovedisplayskip 20.17\p@ \@plus 5.12\p@ \@minus 11.02\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 4.12\p@%
        \belowdisplayshortskip 10.86\p@ \@plus 4.96\p@ \@minus 5.90\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 10.86\p@ \@plus 4.19\p@ \@minus 4.19\p@%
            \parsep 5.90\p@ \@plus 3.34\p@ \@minus 2.56\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{25.40}{30.98}%
        \abovedisplayskip 24.97\p@ \@plus 7.68\p@ \@minus 12.64\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 6.75\p@%
        \belowdisplayshortskip 13.88\p@ \@plus 7.13\p@ \@minus 6.75\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 15.82\p@ \@plus 5.90\p@ \@minus 7.45\p@%
            \parsep 7.91\p@ \@plus 4.19\p@ \@minus 2.56\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{27.96}{33.94}%
        \abovedisplayskip 28.00\p@ \@plus 6.75\p@ \@minus 15.20\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 7.68\p@%
        \belowdisplayshortskip 16.17\p@ \@plus 8.49\p@ \@minus 7.68\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{32.58}{39.66}}%
    \renewcommand\Large{\@setfontsize\Large{38.88}{49.18}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{45.85}{58.64}}%
    \renewcommand\huge{\@setfontsize\huge{55.08}{67.88}}%
    \renewcommand\Huge{\@setfontsize\Huge{63.99}{76.79}}%
    %
    \setlength\smallskipamount{7.68\p@ \@plus 0.99\p@ \@minus 1.70\p@}%
    \setlength\medskipamount{15.36\p@ \@plus 5.12\p@ \@minus 5.12\p@}%
    \setlength\bigskipamount{30.72\p@ \@plus 10.24\p@ \@minus 10.24\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{30.72\p@}%
        \setlength\headsep{0.68in}%
        \setlength\topskip{28.00\p@}%
        \setlength\footskip{0.97in}%
        \setlength\@tempdimb{930.79\p@}%
    \fi%
    %
    \setlength\marginparsep{4.05\p@}%
    \setlength\marginparpush{14.35\p@}%
    %
    \setlength\footnotesep{19.27\p@}%
    \setlength{\skip\footins}{25.28\p@ \@plus 10.24\p@ \@minus 5.12\p@}%
    %
    \setlength\floatsep{30.72\p@ \@plus 5.12\p@ \@minus 6.67\p@}%
    \setlength\textfloatsep{51.19\p@ \@plus 5.12\p@ \@minus 10.24\p@}%
    \setlength\intextsep{32.27\p@ \@plus 6.67\p@ \@minus 6.67\p@}%
    %
    \setlength\dblfloatsep{32.27\p@ \@plus 5.12\p@ \@minus 6.67\p@}%
    \setlength\dbltextfloatsep{51.19\p@ \@plus 5.12\p@ \@minus 10.24\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 2.56fil}%
    \setlength\@fpsep{22.03\p@ \@plus 5.12fil}%
    \setlength\@fpbot{0.00\p@ \@plus 2.56fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 2.56fil}%
    \setlength\@dblfpsep{22.03\p@ \@plus 5.12fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 2.56fil}%
    %
    \setlength\partopsep{6.75\p@ \@plus 3.34\p@ \@minus 3.34\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 11.44\p@ \@plus 5.51\p@ \@minus 2.56\p@%
        \topsep 22.88\p@ \@plus 7.52\p@ \@minus 12.64\p@%
        \itemsep11.44\p@ \@plus 5.51\p@ \@minus 2.56\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 11.44\p@ \@plus 5.51\p@ \@minus 2.56\p@%
        \parsep 5.51\p@ \@plus 2.56\p@ \@minus 2.56\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 5.51\p@ \@plus 2.56\p@\@minus 2.56\p@%
        \parsep 0.00\p@%
        \partopsep 2.56\p@ \@plus 0.00\p@ \@minus 2.56\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 32 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXXLII}{%
    \renewcommand\tiny{\@setfontsize\tiny{16.48}{19.41}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{22.34}{26.19}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{26.15}{31.42}%
        \abovedisplayskip 23.05\p@ \@plus 5.85\p@ \@minus 12.59\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 4.70\p@%
        \belowdisplayshortskip 12.41\p@ \@plus 5.67\p@ \@minus 6.74\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 12.41\p@ \@plus 4.78\p@ \@minus 4.78\p@%
            \parsep 6.74\p@ \@plus 3.81\p@ \@minus 2.93\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{29.03}{35.41}%
        \abovedisplayskip 28.54\p@ \@plus 8.78\p@ \@minus 14.45\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 7.71\p@%
        \belowdisplayshortskip 15.86\p@ \@plus 8.15\p@ \@minus 7.71\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 18.08\p@ \@plus 6.74\p@ \@minus 8.52\p@%
            \parsep 9.04\p@ \@plus 4.78\p@ \@minus 2.93\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{31.95}{38.79}%
        \abovedisplayskip 32.00\p@ \@plus 7.71\p@ \@minus 17.37\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 8.78\p@%
        \belowdisplayshortskip 18.48\p@ \@plus 9.71\p@ \@minus 8.78\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{37.24}{45.32}}%
    \renewcommand\Large{\@setfontsize\Large{44.43}{56.21}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{52.40}{67.02}}%
    \renewcommand\huge{\@setfontsize\huge{62.95}{77.58}}%
    \renewcommand\Huge{\@setfontsize\Huge{73.13}{87.76}}%
    %
    \setlength\smallskipamount{8.78\p@ \@plus 0.86\p@ \@minus 1.49\p@}%
    \setlength\medskipamount{17.55\p@ \@plus 5.85\p@ \@minus 5.85\p@}%
    \setlength\bigskipamount{35.10\p@ \@plus 11.70\p@ \@minus 11.70\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{35.10\p@}%
        \setlength\headsep{0.78in}%
        \setlength\topskip{32.00\p@}%
        \setlength\footskip{1.11in}%
        \setlength\@tempdimb{1063.76\p@}%
    \fi%
    %
    \setlength\marginparsep{3.54\p@}%
    \setlength\marginparpush{16.40\p@}%
    %
    \setlength\footnotesep{22.03\p@}%
    \setlength{\skip\footins}{28.90\p@ \@plus 11.70\p@ \@minus 5.85\p@}%
    %
    \setlength\floatsep{35.10\p@ \@plus 5.85\p@ \@minus 7.63\p@}%
    \setlength\textfloatsep{58.51\p@ \@plus 5.85\p@ \@minus 11.70\p@}%
    \setlength\intextsep{36.88\p@ \@plus 7.63\p@ \@minus 7.63\p@}%
    %
    \setlength\dblfloatsep{36.88\p@ \@plus 5.85\p@ \@minus 7.63\p@}%
    \setlength\dbltextfloatsep{58.51\p@ \@plus 5.85\p@ \@minus 11.70\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 2.93fil}%
    \setlength\@fpsep{25.18\p@ \@plus 5.85fil}%
    \setlength\@fpbot{0.00\p@ \@plus 2.93fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 2.93fil}%
    \setlength\@dblfpsep{25.18\p@ \@plus 5.85fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 2.93fil}%
    %
    \setlength\partopsep{7.71\p@ \@plus 3.81\p@ \@minus 3.81\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 13.07\p@ \@plus 6.29\p@ \@minus 2.93\p@%
        \topsep 26.15\p@ \@plus 8.60\p@ \@minus 14.45\p@%
        \itemsep13.07\p@ \@plus 6.29\p@ \@minus 2.93\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 13.07\p@ \@plus 6.29\p@ \@minus 2.93\p@%
        \parsep 6.29\p@ \@plus 2.93\p@ \@minus 2.93\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 6.29\p@ \@plus 2.93\p@\@minus 2.93\p@%
        \parsep 0.00\p@%
        \partopsep 2.93\p@ \@plus 0.00\p@ \@minus 2.93\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 36 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXXLVI}{%
    \renewcommand\tiny{\@setfontsize\tiny{18.55}{21.84}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{25.13}{29.46}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{29.42}{35.35}%
        \abovedisplayskip 25.93\p@ \@plus 6.58\p@ \@minus 14.16\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 5.29\p@%
        \belowdisplayshortskip 13.96\p@ \@plus 6.38\p@ \@minus 7.58\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 13.96\p@ \@plus 5.38\p@ \@minus 5.38\p@%
            \parsep 7.58\p@ \@plus 4.29\p@ \@minus 3.29\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{32.66}{39.83}%
        \abovedisplayskip 32.11\p@ \@plus 9.87\p@ \@minus 16.25\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 8.67\p@%
        \belowdisplayshortskip 17.85\p@ \@plus 9.17\p@ \@minus 8.67\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 20.35\p@ \@plus 7.58\p@ \@minus 9.58\p@%
            \parsep 10.17\p@ \@plus 5.38\p@ \@minus 3.29\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{35.95}{43.64}%
        \abovedisplayskip 36.00\p@ \@plus 8.67\p@ \@minus 19.55\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 9.87\p@%
        \belowdisplayshortskip 20.79\p@ \@plus 10.92\p@ \@minus 9.87\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{41.89}{50.99}}%
    \renewcommand\Large{\@setfontsize\Large{49.99}{63.24}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{58.95}{75.40}}%
    \renewcommand\huge{\@setfontsize\huge{70.82}{87.27}}%
    \renewcommand\Huge{\@setfontsize\Huge{82.27}{98.73}}%
    %
    \setlength\smallskipamount{9.87\p@ \@plus 0.77\p@ \@minus 1.32\p@}%
    \setlength\medskipamount{19.75\p@ \@plus 6.58\p@ \@minus 6.58\p@}%
    \setlength\bigskipamount{39.49\p@ \@plus 13.16\p@ \@minus 13.16\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{39.49\p@}%
        \setlength\headsep{0.88in}%
        \setlength\topskip{36.00\p@}%
        \setlength\footskip{1.25in}%
        \setlength\@tempdimb{1196.73\p@}%
    \fi%
    %
    \setlength\marginparsep{3.15\p@}%
    \setlength\marginparpush{18.45\p@}%
    %
    \setlength\footnotesep{24.78\p@}%
    \setlength{\skip\footins}{32.51\p@ \@plus 13.16\p@ \@minus 6.58\p@}%
    %
    \setlength\floatsep{39.49\p@ \@plus 6.58\p@ \@minus 8.58\p@}%
    \setlength\textfloatsep{65.82\p@ \@plus 6.58\p@ \@minus 13.16\p@}%
    \setlength\intextsep{41.49\p@ \@plus 8.58\p@ \@minus 8.58\p@}%
    %
    \setlength\dblfloatsep{41.49\p@ \@plus 6.58\p@ \@minus 8.58\p@}%
    \setlength\dbltextfloatsep{65.82\p@ \@plus 6.58\p@ \@minus 13.16\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 3.29fil}%
    \setlength\@fpsep{28.33\p@ \@plus 6.58fil}%
    \setlength\@fpbot{0.00\p@ \@plus 3.29fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 3.29fil}%
    \setlength\@dblfpsep{28.33\p@ \@plus 6.58fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 3.29fil}%
    %
    \setlength\partopsep{8.67\p@ \@plus 4.29\p@ \@minus 4.29\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 14.71\p@ \@plus 7.08\p@ \@minus 3.29\p@%
        \topsep 29.42\p@ \@plus 9.67\p@ \@minus 16.25\p@%
        \itemsep14.71\p@ \@plus 7.08\p@ \@minus 3.29\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 14.71\p@ \@plus 7.08\p@ \@minus 3.29\p@%
        \parsep 7.08\p@ \@plus 3.29\p@ \@minus 3.29\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 7.08\p@ \@plus 3.29\p@\@minus 3.29\p@%
        \parsep 0.00\p@%
        \partopsep 3.29\p@ \@plus 0.00\p@ \@minus 3.29\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 40 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXL}{%
    \renewcommand\tiny{\@setfontsize\tiny{20.61}{24.26}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{27.92}{32.74}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{32.69}{39.27}%
        \abovedisplayskip 28.81\p@ \@plus 7.31\p@ \@minus 15.74\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 5.88\p@%
        \belowdisplayshortskip 15.52\p@ \@plus 7.09\p@ \@minus 8.42\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 15.52\p@ \@plus 5.98\p@ \@minus 5.98\p@%
            \parsep 8.42\p@ \@plus 4.77\p@ \@minus 3.66\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{36.29}{44.26}%
        \abovedisplayskip 35.68\p@ \@plus 10.97\p@ \@minus 18.06\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 9.64\p@%
        \belowdisplayshortskip 19.83\p@ \@plus 10.19\p@ \@minus 9.64\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 22.61\p@ \@plus 8.42\p@ \@minus 10.65\p@%
            \parsep 11.30\p@ \@plus 5.98\p@ \@minus 3.66\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{39.94}{48.48}%
        \abovedisplayskip 40.00\p@ \@plus 9.64\p@ \@minus 21.72\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 10.97\p@%
        \belowdisplayshortskip 23.10\p@ \@plus 12.13\p@ \@minus 10.97\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{46.55}{56.65}}%
    \renewcommand\Large{\@setfontsize\Large{55.54}{70.26}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{65.49}{83.78}}%
    \renewcommand\huge{\@setfontsize\huge{78.69}{96.97}}%
    \renewcommand\Huge{\@setfontsize\Huge{91.41}{109.70}}%
    %
    \setlength\smallskipamount{10.97\p@ \@plus 0.69\p@ \@minus 1.19\p@}%
    \setlength\medskipamount{21.94\p@ \@plus 7.31\p@ \@minus 7.31\p@}%
    \setlength\bigskipamount{43.88\p@ \@plus 14.63\p@ \@minus 14.63\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{43.88\p@}%
        \setlength\headsep{0.97in}%
        \setlength\topskip{40.00\p@}%
        \setlength\footskip{1.39in}%
        \setlength\@tempdimb{1329.70\p@}%
    \fi%
    %
    \setlength\marginparsep{2.83\p@}%
    \setlength\marginparpush{20.51\p@}%
    %
    \setlength\footnotesep{27.53\p@}%
    \setlength{\skip\footins}{36.12\p@ \@plus 14.63\p@ \@minus 7.31\p@}%
    %
    \setlength\floatsep{43.88\p@ \@plus 7.31\p@ \@minus 9.54\p@}%
    \setlength\textfloatsep{73.13\p@ \@plus 7.31\p@ \@minus 14.63\p@}%
    \setlength\intextsep{46.10\p@ \@plus 9.54\p@ \@minus 9.54\p@}%
    %
    \setlength\dblfloatsep{46.10\p@ \@plus 7.31\p@ \@minus 9.54\p@}%
    \setlength\dbltextfloatsep{73.13\p@ \@plus 7.31\p@ \@minus 14.63\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 3.66fil}%
    \setlength\@fpsep{31.47\p@ \@plus 7.31fil}%
    \setlength\@fpbot{0.00\p@ \@plus 3.66fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 3.66fil}%
    \setlength\@dblfpsep{31.47\p@ \@plus 7.31fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 3.66fil}%
    %
    \setlength\partopsep{9.64\p@ \@plus 4.77\p@ \@minus 4.77\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 16.34\p@ \@plus 7.87\p@ \@minus 3.66\p@%
        \topsep 32.69\p@ \@plus 10.75\p@ \@minus 18.06\p@%
        \itemsep16.34\p@ \@plus 7.87\p@ \@minus 3.66\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 16.34\p@ \@plus 7.87\p@ \@minus 3.66\p@%
        \parsep 7.87\p@ \@plus 3.66\p@ \@minus 3.66\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 7.87\p@ \@plus 3.66\p@\@minus 3.66\p@%
        \parsep 0.00\p@%
        \partopsep 3.66\p@ \@plus 0.00\p@ \@minus 3.66\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
% Font size 44 (scaled from standard article-class)
\newcommand{\AMfontSetDocumentFontSizeXLIV}{%
    \renewcommand\tiny{\@setfontsize\tiny{22.67}{26.69}}%
    \renewcommand\scriptsize{\@setfontsize\scriptsize{30.71}{36.01}}%
    \renewcommand\footnotesize{%
        \@setfontsize\footnotesize{35.96}{43.20}%
        \abovedisplayskip 31.69\p@ \@plus 8.04\p@ \@minus 17.31\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 6.47\p@%
        \belowdisplayshortskip 17.07\p@ \@plus 7.80\p@ \@minus 9.27\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 17.07\p@ \@plus 6.58\p@ \@minus 6.58\p@%
            \parsep 9.27\p@ \@plus 5.24\p@ \@minus 4.02\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\small{%
        \@setfontsize\small{39.92}{48.68}%
        \abovedisplayskip 39.24\p@ \@plus 12.07\p@ \@minus 19.87\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 10.60\p@%
        \belowdisplayshortskip 21.81\p@ \@plus 11.21\p@ \@minus 10.60\p@%
        \belowdisplayskip \abovedisplayskip%
        \def\@listi{%
            \leftmargin\leftmargini%
            \topsep 24.87\p@ \@plus 9.27\p@ \@minus 11.71\p@%
            \parsep 12.43\p@ \@plus 6.58\p@ \@minus 4.02\p@%
            \itemsep \parsep%
        }%
    }%
    \renewcommand\normalsize{%
        \@setfontsize\normalsize{43.93}{53.33}%
        \abovedisplayskip 44.00\p@ \@plus 10.60\p@ \@minus 23.89\p@%
        \abovedisplayshortskip 0.00\p@ \@plus 12.07\p@%
        \belowdisplayshortskip 25.41\p@ \@plus 13.34\p@ \@minus 12.07\p@%
        \belowdisplayskip \abovedisplayskip%
        \let\@listi\@listI%
    }%
    \renewcommand\large{\@setfontsize\large{51.20}{62.32}}%
    \renewcommand\Large{\@setfontsize\Large{61.10}{77.29}}%
    \renewcommand\LARGE{\@setfontsize\LARGE{72.04}{92.16}}%
    \renewcommand\huge{\@setfontsize\huge{86.56}{106.67}}%
    \renewcommand\Huge{\@setfontsize\Huge{100.56}{120.67}}%
    %
    \setlength\smallskipamount{12.07\p@ \@plus 0.63\p@ \@minus 1.08\p@}%
    \setlength\medskipamount{24.13\p@ \@plus 8.04\p@ \@minus 8.04\p@}%
    \setlength\bigskipamount{48.27\p@ \@plus 16.09\p@ \@minus 16.09\p@}%
    %
    \ifdefined\optAMbeamer\else%
        \setlength\headheight{48.27\p@}%
        \setlength\headsep{1.07in}%
        \setlength\topskip{44.00\p@}%
        \setlength\footskip{1.53in}%
        \setlength\@tempdimb{1462.67\p@}%
    \fi%
    %
    \setlength\marginparsep{2.58\p@}%
    \setlength\marginparpush{22.56\p@}%
    %
    \setlength\footnotesep{30.29\p@}%
    \setlength{\skip\footins}{39.73\p@ \@plus 16.09\p@ \@minus 8.04\p@}%
    %
    \setlength\floatsep{48.27\p@ \@plus 8.04\p@ \@minus 10.49\p@}%
    \setlength\textfloatsep{80.44\p@ \@plus 8.04\p@ \@minus 16.09\p@}%
    \setlength\intextsep{50.71\p@ \@plus 10.49\p@ \@minus 10.49\p@}%
    %
    \setlength\dblfloatsep{50.71\p@ \@plus 8.04\p@ \@minus 10.49\p@}%
    \setlength\dbltextfloatsep{80.44\p@ \@plus 8.04\p@ \@minus 16.09\p@}%
    %
    \setlength\@fptop{0.00\p@ \@plus 4.02fil}%
    \setlength\@fpsep{34.62\p@ \@plus 8.04fil}%
    \setlength\@fpbot{0.00\p@ \@plus 4.02fil}%
    %
    \setlength\@dblfptop{0.00\p@ \@plus 4.02fil}%
    \setlength\@dblfpsep{34.62\p@ \@plus 8.04fil}%
    \setlength\@dblfpbot{0.00\p@ \@plus 4.02fil}%
    %
    \setlength\partopsep{10.60\p@ \@plus 5.24\p@ \@minus 5.24\p@}%
    %
    \def\@listi{%
        \leftmargin\leftmargini%
        \parsep 17.98\p@ \@plus 8.66\p@ \@minus 4.02\p@%
        \topsep 35.96\p@ \@plus 11.82\p@ \@minus 19.87\p@%
        \itemsep17.98\p@ \@plus 8.66\p@ \@minus 4.02\p@%
    }%
    \let\@listI\@listi%
    \@listi%
    %
    \def\@listii{%
        \leftmargin\leftmarginii%
        \labelwidth\leftmarginii%
        \advance\labelwidth-\labelsep%
        \topsep 17.98\p@ \@plus 8.66\p@ \@minus 4.02\p@%
        \parsep 8.66\p@ \@plus 4.02\p@ \@minus 4.02\p@%
        \itemsep \parsep%
    }%
    %
    \def\@listiii{%
        \leftmargin\leftmarginiii%
        \labelwidth\leftmarginiii%
        \advance\labelwidth-\labelsep%
        \topsep 8.66\p@ \@plus 4.02\p@\@minus 4.02\p@%
        \parsep 0.00\p@%
        \partopsep 4.02\p@ \@plus 0.00\p@ \@minus 4.02\p@%
        \itemsep \topsep%
    }%
    %
    \def\@listiv{%
        \leftmargin\leftmarginiv%
        \labelwidth\leftmarginiv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listv{%
        \leftmargin\leftmarginv%
        \labelwidth\leftmarginv%
        \advance\labelwidth-\labelsep%
    }%
    %
    \def\@listvi{%
        \leftmargin\leftmarginvi%
        \labelwidth\leftmarginvi%
        \advance\labelwidth-\labelsep%
    }%
    %
    \normalsize% Set normalsize as default font size
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMfont.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if module AMfont is enabled
%###################################################################################################
%--- END MODULE AMfont -----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMlang ---------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMlang% Start-check if module AMlang is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMlang.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%===============================
%--- Default Option handling ---
%===============================
% Use german as default, if no language has been selected
\ifdefined\optGerman\else%
\ifdefined\optEnglish\else%
    \def\optGerman{x}%
\fi%
\fi%
%
%
%================
%--- Packages ---
%================
\ifdefined\optGerman%
    \PassOptionsToPackage{english,ngerman}{babel}%
\else\fi%
\ifdefined\optEnglish%
    \PassOptionsToPackage{ngerman,english}{babel}%
\else\fi%
\RequirePackage{babel}%
\RequirePackage{csquotes}%
%
%
%================
%--- Commands ---
%================
% Switch language
% ---------------
\newcommand{\AMlangSwitchToGerman}{%
    \def\optGerman{x}%
    \undef\optEnglish%
    \selectlanguage{ngerman}%
}%
\newcommand{\AMlangSwitchToEnglish}{%
    \undef\optGerman%
    \def\optEnglish{x}%
    \selectlanguage{english}%
}%
%
% Auto-select language
% --------------------
\newcommand{\AMlangGerEng}[2]{\ifdefined\optGerman#1\else\fi\ifdefined\optEnglish#2\else\fi}%
\newcommand{\AMlangForceGerman}[1]{%
    \ifdefined\optEnglish%
        \AMlangSwitchToGerman%
        #1%
        \AMlangSwitchToEnglish%
    \else\fi%
    \ifdefined\optGerman%
        #1%
    \else\fi%
}%
\newcommand{\AMlangForceEnglish}[1]{%
    \ifdefined\optGerman%
        \AMlangSwitchToEnglish%
        #1%
        \AMlangSwitchToGerman%
    \else\fi%
    \ifdefined\optEnglish%
        #1%
    \else\fi%
}%
%
%
%===================
%--- Definitions ---
%===================
% TUM
% ---
\def\AMlangTUM{\AMlangGerEng{Technische Universität München}{Technical University of Munich}}%
%
% Schools
% -----------
\def\AMlangSchoolMW{\AMlangGerEng{TUM School of Engineering and Design}{TUM School of Engineering and Design}}%
\def\AMlangSchoolED{\AMlangGerEng{TUM School of Engineering and Design}{TUM School of Engineering and Design}}%
\def\AMlangSchoolIN{\AMlangGerEng{Fakultät für Informatik}{School of Informatics}}%
\def\AMlangSchoolCIT{\AMlangGerEng{TUM School of Computation, Information and Technology}{TUM School of Computation, Information and Technology}}%
%
% Current school
% ------------------
\def\AMlangSchool{\AMlangSchoolED}%
%
% Chairs
% ------
% TUM School of Engineering and Design
\def\AMlangChairEDAER{\AMlangGerEng{Lehrstuhl für Aerodynamik und Strömungsmechanik}{Chair of Aerodynamics and Fluid Mechanics}}%
\def\AMlangChairEDVIB{\AMlangGerEng{Lehrstuhl für Akustik mobiler Systeme}{Chair of Vibroacoustics of Vehicles and Machines}}%
\def\AMlangChairEDAM{\AMlangGerEng{Lehrstuhl für Angewandte Mechanik}{Chair of Applied Mechanics}}%
\def\AMlangChairEDAPT{\AMlangGerEng{Lehrstuhl für Anlagen- und Prozesstechnik}{Chair of Plant and Process Technology}}%
\def\AMlangChairEDAIS{\AMlangGerEng{Lehrstuhl für Automatisierung und Informationssysteme}{Chair of Automation and Information Systems}}%
\def\AMlangChairEDIWB{\AMlangGerEng{Institut für Werkzeugmaschinen und Betriebswissenschaften (iwb)}{Institute for Machine Tools and Industrial Management (iwb)}}%
\def\AMlangChairEDIWBLWF{\AMlangGerEng{Lehrstuhl für Werkzeugmaschinen und Fertigungstechnik}{Chair of Machine Tools and Manufacturing Technology}}%
\def\AMlangChairEDIWBLBM{\AMlangGerEng{Lehrstuhl für Betriebswissenschaften und Montagetechnik}{Chair of Industrial Management and Assembly Technologies}}%
\def\AMlangChairEDIWBPE{\AMlangGerEng{Lehrstuhl für Produktionstechnik und Energiespeichersysteme}{Chair for Production Technology and Energy Storage Systems}}%
\def\AMlangChairEDBVT{\AMlangGerEng{Lehrstuhl für Bioverfahrenstechnik}{Chair of Biochemical Engineering}}%
\def\AMlangChairEDLCC{\AMlangGerEng{Lehrstuhl für Carbon Composites}{Chair of Carbon Composites}}%
\def\AMlangChairEDLES{\AMlangGerEng{Lehrstuhl für Energiesysteme}{Chair of Energy Systems}}%
\def\AMlangChairEDLFE{\AMlangGerEng{Lehrstuhl für Ergonomie}{Chair of Ergonomics}}%
\def\AMlangChairEDFTM{\AMlangGerEng{Lehrstuhl für Fahrzeugtechnik}{Chair of Automotive Technology}}%
\def\AMlangChairEDFSD{\AMlangGerEng{Lehrstuhl für Flugsystemdynamik}{Chair of Flight System Dynamics}}%
\def\AMlangChairEDFML{\AMlangGerEng{Lehrstuhl für Fördertechnik Materialfluss Logistik / Technische Logistik}{Chair of Materials Handling, Material Flow, Logistics / Logistics Engineering}}%
\def\AMlangChairEDLHT{\AMlangGerEng{Lehrstuhl für Hubschraubertechnologie}{Chair of Helicopter Technology}}%
\def\AMlangChairEDPKM{\AMlangGerEng{Professur für Kontinuumsmechanik}{Associate Professorship of Continuum Mechanics}}%
\def\AMlangChairEDLLB{\AMlangGerEng{Lehrstuhl für Leichtbau}{Chair of Lightweight Structures}}%
\def\AMlangChairEDLLS{\AMlangGerEng{Lehrstuhl für Luftfahrtsysteme}{Chair of Aircraft Design}}%
\def\AMlangChairEDFZG{\AMlangGerEng{Lehrstuhl für Maschinenelemente}{Chair of Machine Elements}}%
\def\AMlangChairEDMHPC{\AMlangGerEng{Professur für Mechanik auf Höchstleistungsrechnern}{Associate Professorship of Mechanics and High Performance Computing}}%
\def\AMlangChairEDLMT{\AMlangGerEng{Lehrstuhl für Medizintechnik}{Chair of Medical Engineering}}%
\def\AMlangChairEDLMM{\AMlangGerEng{Lehrstuhl für Mikrotechnik und Medizingerätetechnik}{Chair of Micro Technology and Medical Device Technology}}%
\def\AMlangChairEDLNT{\AMlangGerEng{Lehrstuhl für Nukleartechnik}{Chair of Nuclear Technology}}%
\def\AMlangChairEDLNM{\AMlangGerEng{Lehrstuhl für Numerische Mechanik}{Chair of Computational Mechanics}}%
\def\AMlangChairEDPMW{\AMlangGerEng{Professur für Plasma-Material-Wechselwirkung}{Associate Professorship of Plasma Material Interaction}}%
\def\AMlangChairEDPE{\AMlangGerEng{Lehrstuhl für Produktentwicklung}{Chair of Product Development}}%
\def\AMlangChairEDPTM{\AMlangGerEng{Lehrstuhl für Produktion und Technik in der Medienbranche}{Chair of Production and Technology in Media}}%
\def\AMlangChairEDLFA{\AMlangGerEng{Lehrstuhl für Raumfahrtantriebe}{Chair of Space Propulsion}}%
\def\AMlangChairEDLRT{\AMlangGerEng{Lehrstuhl für Raumfahrttechnik}{Chair of Astronautics}}%
\def\AMlangChairEDRT{\AMlangGerEng{Lehrstuhl für Regelungstechnik}{Chair of Automatic Control}}%
\def\AMlangChairEDSES{\AMlangGerEng{Professur für Sichere Eingebettete Systeme}{Assistant Professorship of Safe Embedded Systems}}%
\def\AMlangChairEDSPGM{\AMlangGerEng{Professur für Sportgeräte und Sportmaterialien}{Associate Professorship of Sport Equipment and Materials}}%
\def\AMlangChairEDTD{\AMlangGerEng{Lehrstuhl für Thermodynamik}{Chair of Thermodynamics}}%
\def\AMlangChairEDTFD{\AMlangGerEng{Professur für Thermofluiddynamik}{Associate Professorship of Thermo-Fluid Dynamics}}%
\def\AMlangChairEDUTG{\AMlangGerEng{Lehrstuhl für Umformtechnik und Gießereiwesen}{Chair of Metal Forming and Casting}}%
\def\AMlangChairEDLVK{\AMlangGerEng{Lehrstuhl für Verbrennungskraftmaschinen}{Chair of Internal Combustion Engines}}%
\def\AMlangChairEDWKM{\AMlangGerEng{Lehrstuhl für Werkstoffkunde und Werkstoffmechanik}{Chair of Materials Science and Mechanics of Materials}}%
\def\AMlangChairEDLWE{\AMlangGerEng{Lehrstuhl für Windenergie}{Chair of Wind Energy}}%
\def\AMlangChairEDZFP{\AMlangGerEng{Lehrstuhl für Zerstörungsfreie Prüfung}{Chair of Non-Destructive Testing}}%
\def\AMlangChairEDZL{\AMlangGerEng{Zentrale Lehre}{Central Teaching Unit}}%
\def\AMlangChairEDGEN{\AMlangGerEng{Generic Chair}{Generic Chair}}%
%
% TUM School of Computation, Information and Technology
\def\AMlangChairCITAIR{\AMlangGerEng{Lehrstuhl für Robotik, Künstliche Intelligenz und Echtzeitsysteme}{Chair of Robotics, Artificial Intelligence and Real-time Systems}}%
%
% School of Informatics
\def\AMlangChairINAIR{\AMlangChairCITAIR}%
\def\AMlangChairINSSE{\AMlangGerEng{Lehrstuhl für Software und Systems Engineering}{Chair of Software and Systems Engineering}}%
%
% Current chair
% -------------
\def\AMlangChair{\AMlangChairEDAM}%
%
% Roles
% -----
\def\AMlangresearchassistantMale{\AMlangGerEng{wissenschaftlicher Mitarbeiter}{research assistant}}%
\def\AMlangResearchassistantMale{\AMlangGerEng{Wissenschaftlicher Mitarbeiter}{Research assistant}}%
\def\AMlangResearchAssistantMale{\AMlangGerEng{Wissenschaftlicher Mitarbeiter}{Research Assistant}}%
\def\AMlangresearchassistantFemale{\AMlangGerEng{wissenschaftliche Mitarbeiterin}{research assistant}}%
\def\AMlangResearchassistantFemale{\AMlangGerEng{Wissenschaftliche Mitarbeiterin}{Research assistant}}%
\def\AMlangResearchAssistantFemale{\AMlangGerEng{Wissenschaftliche Mitarbeiterin}{Research Assistant}}%
\def\AMlangtenuredprofessorMale{\AMlangGerEng{Ordinarius}{tenured professor}}%
\def\AMlangTenuredprofessorMale{\AMlangGerEng{Ordinarius}{Tenured professor}}%
\def\AMlangTenuredProfessorMale{\AMlangGerEng{Ordinarius}{Tenured Professor}}%
\def\AMlangtenuredprofessorFemale{\AMlangGerEng{Ordinaria}{tenured professor}}%
\def\AMlangTenuredprofessorFemale{\AMlangGerEng{Ordinaria}{Tenured professor}}%
\def\AMlangTenuredProfessorFemale{\AMlangGerEng{Ordinaria}{Tenured Professor}}%
\def\AMlangsupervisor{\AMlangGerEng{Betreuer}{supervisor}}%
\def\AMlangSupervisor{\AMlangGerEng{Betreuer}{Supervisor}}%
\def\AMlangsupervision{\AMlangGerEng{Betreuung}{supervision}}%
\def\AMlangSupervision{\AMlangGerEng{Betreuung}{Supervision}}%
\def\AMlangexaminer{\AMlangGerEng{Prüfer}{examiner}}%
\def\AMlangExaminer{\AMlangGerEng{Prüfer}{Examiner}}%
%
% Student thesis
% --------------
\def\AMlangbachelorsthesis{\AMlangGerEng{Bachelorarbeit}{bachelor's thesis}}%
\def\AMlangBachelorsthesis{\AMlangGerEng{Bachelorarbeit}{Bachelor's thesis}}%
\def\AMlangBachelorsThesis{\AMlangGerEng{Bachelorarbeit}{Bachelor's Thesis}}%
\def\AMlangdiplomathesis{\AMlangGerEng{Diplomarbeit}{diploma thesis}}%
\def\AMlangDiplomathesis{\AMlangGerEng{Diplomarbeit}{Diploma thesis}}%
\def\AMlangDiplomaThesis{\AMlangGerEng{Diplomarbeit}{Diploma Thesis}}%
\def\AMlangsemesterthesis{\AMlangGerEng{Semesterarbeit}{semester thesis}}%
\def\AMlangSemesterthesis{\AMlangGerEng{Semesterarbeit}{Semester thesis}}%
\def\AMlangSemesterThesis{\AMlangGerEng{Semesterarbeit}{Semester Thesis}}%
\def\AMlangmastersthesis{\AMlangGerEng{Masterarbeit}{master's thesis}}%
\def\AMlangMastersthesis{\AMlangGerEng{Masterarbeit}{Master's thesis}}%
\def\AMlangMastersThesis{\AMlangGerEng{Masterarbeit}{Master's Thesis}}%
\def\AMlanginterdisciplinaryproject{\AMlangGerEng{interdisziplinäres Projekt}{interdisciplinary project}}%
\def\AMlangInterdisciplinaryproject{\AMlangGerEng{Interdisziplinäres Projekt}{Interdisciplinary project}}%
\def\AMlangInterdisciplinaryProject{\AMlangGerEng{Interdisziplinäres Projekt}{Interdisciplinary Project}}%
\def\AMlangScientificWorkForObtainingAcademicDegree{\AMlangGerEng{Wissenschaftliche Arbeit zur Erlangung des Grades}{Scientific work for obtaining the academic degree}}%
%
% Fields of Study
% --------------
\def\AMlangFieldOfStudyinformatics{\AMlangGerEng{Informatik}{informatics}}%
\def\AMlangFieldOfStudyInformatics{\AMlangGerEng{Informatik}{Informatics}}%
\def\AMlangFieldOfStudygamesengineering{\AMlangGerEng{Games Engineering}{games engineering}}%
\def\AMlangFieldOfStudyGamesengineering{\AMlangGerEng{Games Engineering}{Games engineering}}%
\def\AMlangFieldOfStudyGamesEngineering{\AMlangGerEng{Games Engineering}{Games Engineering}}%
\def\AMlangFieldOfStudyinformationsystems{\AMlangGerEng{Wirtschaftsinformatik}{information systems}}%
\def\AMlangFieldOfStudyInformationsystems{\AMlangGerEng{Wirtschaftsinformatik}{Information systems}}%
\def\AMlangFieldOfStudyInformationSystems{\AMlangGerEng{Wirtschaftsinformatik}{Information Systems}}%
\def\AMlangFieldOfStudybiomedicalcomputing{\AMlangGerEng{Biomedical Computing}{biomedical computing}}%
\def\AMlangFieldOfStudyBiomedicalcomputing{\AMlangGerEng{Biomedical Computing}{Biomedical computing}}%
\def\AMlangFieldOfStudyBiomedicalComputing{\AMlangGerEng{Biomedical Computing}{Biomedical Computing}}%
\def\AMlangFieldOfStudyautomotivesoftwareengineering{\AMlangGerEng{Automotive Software Engineering}{automotive software engineering}}%
\def\AMlangFieldOfStudyAutomotivesoftwareengineering{\AMlangGerEng{Automotive Software Engineering}{Automotive software engineering}}%
\def\AMlangFieldOfStudyAutomotiveSoftwareEngineering{\AMlangGerEng{Automotive Software Engineering}{Automotive Software Engineering}}%
\def\AMlangFieldOfstudyroboticscognitionintelligence{\AMlangGerEng{Robotics, Cognition, Intelligence}{robotics, cognition, intelligence}}%
\def\AMlangFieldOfStudyRoboticscognitionintelligence{\AMlangGerEng{Robotics, Cognition, Intelligence}{Robotics, cognition, intelligence}}%
\def\AMlangFieldOfStudyRoboticsCognitionIntelligence{\AMlangGerEng{Robotics, Cognition, Intelligence}{Robotics, Cognition, Intelligence}}%
%
% Ph.D. Degrees
% --------------
\def\AMlangPhdDegreeDrIng{Doktors der Ingenieurwissenschaften (Dr.-Ing.)}%
\def\AMlangPhdDegreeDrIngFemale{Doktorin der Ingenieurwissenschaften (Dr.-Ing.)}%
\def\AMlangPhdDegreeDrIngOld{Doktor-Ingenieurs (Dr.-Ing.)}%
\def\AMlangPhdDegreeDrRerNat{Doktors der Naturwissenschaften (Dr.\,rer.\,nat.)}%
\def\AMlangPhdDegreeDrRerNatFemale{Doktorin der Naturwissenschaften (Dr.\,rer.\,nat.)}%
%
% Date and Time
% -------------
\def\AMlangDate{\AMlangGerEng{Datum}{Date}}%
%
\newcommand{\AMlangMonth}[1]{%
    \ifdefined\optGerman\ifcase#1\or Januar\or Februar\or M\"arz\or April\or Mai\or Juni\or Juli\or August\or September\or Oktober\or November\or Dezember\fi\else\fi%
    \ifdefined\optEnglish\ifcase#1\or January\or February\or March\or April\or May\or June\or July\or August\or September\or October\or November\or December\fi\else\fi%
}%
\newcommand{\AMlangMonthShort}[1]{%
    \ifdefined\optGerman\ifcase#1\or Jan\or Feb\or M\"ar\or Apr\or Mai\or Jun\or Jul\or Aug\or Sep\or Okt\or Nov\or Dez\fi\else\fi%
    \ifdefined\optEnglish\ifcase#1\or Jan\or Feb\or Mar\or Apr\or May\or Jun\or Jul\or Aug\or Sep\or Oct\or Nov\or Dec\fi\else\fi%
}%
\newcommand{\AMlangmonth}[1]{%
    \ifdefined\optGerman\ifcase#1\or Januar\or Februar\or M\"arz\or April\or Mai\or Juni\or Juli\or August\or September\or Oktober\or November\or Dezember\fi\else\fi%
    \ifdefined\optEnglish\ifcase#1\or january\or february\or march\or april\or may\or june\or july\or august\or september\or october\or november\or december\fi\else\fi%
}%
\newcommand{\AMlangmonthShort}[1]{%
    \ifdefined\optGerman\ifcase#1\or Jan\or Feb\or M\"ar\or Apr\or Mai\or Jun\or Jul\or Aug\or Sep\or Okt\or Nov\or Dez\fi\else\fi%
    \ifdefined\optEnglish\ifcase#1\or jan\or feb\or mar\or apr\or may\or jun\or jul\or aug\or sep\or oct\or nov\or dec\fi\else\fi%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMlang.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if module AMlang is enabled
%###################################################################################################
%--- END MODULE AMlang -----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMaddress ------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMaddress% Start-check if module AMaddress is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMaddress.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%===================
%--- Definitions ---
%===================
% City
\def\AMaddressCityMunich{\AMlangGerEng{München}{Munich}}%
\def\AMaddressCityGarching{Garching}%
\def\AMaddressCityED{\AMaddressCityGarching}%
\def\AMaddressCityIN{\AMaddressCityGarching}%
\def\AMaddressCityCIT{\AMaddressCityMunich}%
\def\AMaddressCitySchool{\AMaddressCityED}%
\def\AMaddressCityChair{\AMaddressCitySchool}% (uses city of school by default)
%
% Postal Code
\def\AMaddressPostalCodeGarching{85748}%
\def\AMaddressPostalCodeMunich{80333}%
\def\AMaddressPostalCodeED{\AMaddressPostalCodeGarching}%
\def\AMaddressPostalCodeIN{\AMaddressPostalCodeGarching}%
\def\AMaddressPostalCodeCIT{\AMaddressPostalCodeMunich}%
\def\AMaddressPostalCodeSchool{\AMaddressPostalCodeED}%
\def\AMaddressPostalCodeChair{\AMaddressPostalCodeSchool}% (uses postal code of school by default)
%
% Street
\def\AMaddressStreetED{Boltzmannstraße 15}%
\def\AMaddressStreetIN{Boltzmannstraße 3}%
\def\AMaddressStreetCIT{Arcisstraße 21}%
\def\AMaddressStreetSchool{\AMaddressStreetED}%
\def\AMaddressStreetChair{\AMaddressStreetSchool}% (uses street of school by default)
%
% IBAN
\def\AMaddressIBANTUM{DE10700500000000024866}%
\def\AMaddressIBANED{\AMaddressIBANTUM}% (uses IBAN of TUM by default)
\def\AMaddressIBANIN{\AMaddressIBANTUM}% (uses IBAN of TUM by default)
\def\AMaddressIBANCIT{\AMaddressIBANTUM}% (uses IBAN of TUM by default)
\def\AMaddressIBANSchool{\AMaddressIBANED}%
\def\AMaddressIBANChair{\AMaddressIBANSchool}% (uses IBAN of the school by default)
%
% BIC
\def\AMaddressBICTUM{BYLADEMM}%
\def\AMaddressBICED{\AMaddressBICTUM}% (uses BIC of TUM by default)
\def\AMaddressBICIN{\AMaddressBICTUM}% (uses BIC of TUM by default)
\def\AMaddressBICCIT{\AMaddressBICTUM}% (uses BIC of TUM by default)
\def\AMaddressBICSchool{\AMaddressBICED}%
\def\AMaddressBICChair{\AMaddressBICSchool}% (uses BIC of the school by default)
%
% Name of credit institution
\def\AMaddressCreditInstitutionTUM{Bayerische Landesbank}%
\def\AMaddressCreditInstitutionED{\AMaddressCreditInstitutionTUM}% (uses credit institution of TUM by default)
\def\AMaddressCreditInstitutionIN{\AMaddressCreditInstitutionTUM}% (uses credit institution of TUM by default)
\def\AMaddressCreditInstitutionCIT{\AMaddressCreditInstitutionTUM}% (uses credit institution of TUM by default)
\def\AMaddressCreditInstitutionSchool{\AMaddressCreditInstitutionED}%
\def\AMaddressCreditInstitutionChair{\AMaddressCreditInstitutionSchool}% (uses credit institution of the school by default)
%
% Website
\def\AMaddressWebsiteTUM{www.tum.de}%
\def\AMaddressWebsiteED{www.ed.tum.de}%
\def\AMaddressWebsiteIN{www.in.tum.de}%
\def\AMaddressWebsiteCIT{www.cit.tum.de}%
\def\AMaddressWebsiteSchool{\AMaddressWebsiteED}%
\def\AMaddressWebsiteChair{\AMaddressWebsiteSchool/\lowercase{AM}}%
%
% Tax number
\def\AMaddressTaxNumberTUM{143/241/80037}%
\def\AMaddressTaxNumberED{\AMaddressTaxNumberTUM}% (uses tax number of TUM by default)
\def\AMaddressTaxNumberIN{\AMaddressTaxNumberTUM}% (uses tax number of TUM by default)
\def\AMaddressTaxNumberCIT{\AMaddressTaxNumberTUM}% (uses tax number of TUM by default)
\def\AMaddressTaxNumberSchool{\AMaddressTaxNumberED}%
\def\AMaddressTaxNumberChair{\AMaddressTaxNumberSchool}% (uses tax number of the school by default)
%
% Sales tax identification number
\def\AMaddressSalesTaxIDTUM{DE811193231}%
\def\AMaddressSalesTaxIDED{\AMaddressSalesTaxIDTUM}% (uses sales tax number of TUM by default)
\def\AMaddressSalesTaxIDIN{\AMaddressSalesTaxIDTUM}% (uses sales tax number of TUM by default)
\def\AMaddressSalesTaxIDCIT{\AMaddressSalesTaxIDTUM}% (uses sales tax number of TUM by default)
\def\AMaddressSalesTaxIDSchool{\AMaddressSalesTaxIDED}%
\def\AMaddressSalesTaxIDChair{\AMaddressSalesTaxIDSchool}% (uses sales tax identification number of the school by default)
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMaddress.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if module AMaddress is enabled
%###################################################################################################
%--- END MODULE AMaddress --------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMcolor --------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMcolor% Start-check if module AMcolor is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMcolor.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%===============================
%--- Default Option handling ---
%===============================
% Select rgb-model, if no option has been passed
\ifdefined\optRGB\else%
\ifdefined\optCMYK\else%
\ifdefined\optGray\else%
\ifdefined\optMonochrome\else%
    \def\optRGB{x}%
\fi%
\fi%
\fi%
\fi%
%
%
%================
%--- Packages ---
%================
% Select color model
\ifdefined\optRGB%
    \PassOptionsToPackage{rgb}{xcolor}%
\else\fi%
\ifdefined\optCMYK%
    \PassOptionsToPackage{cmyk}{xcolor}%
\else\fi%
\ifdefined\optGray%
    \PassOptionsToPackage{gray}{xcolor}%
\else\fi%
\ifdefined\optMonochrome%
    \PassOptionsToPackage{monochrome}{xcolor}%
\else\fi%
\RequirePackage{xcolor}%
%
%
%===================
%--- Definitions ---
%===================
% TUM colors 
% see https://portal.mytum.de/corporatedesign/index_html/vorlagen/index_farben
% and also see https://portal.mytum.de/corporatedesign/index_html/regeln/index_diagramme
\ifdefined\optCMYK%
    % Primary Colors in CMYK
    \definecolor{TUMBlue}{cmyk}{1,0.43,0,0}%           Pantone 300 C
    \definecolor{TUMWhite}{cmyk}{0,0,0,0}%
    \definecolor{TUMBlack}{cmyk}{0,0,0,1}%
    % Secondary Colors in CMYK
    \definecolor{TUMBlue1}{cmyk}{1,0.57,0.12,0.7}%     Pantone 540
    \definecolor{TUMBlue2}{cmyk}{1,0.54,0.04,0.19}%    Pantone 301
    \definecolor{TUMGray1}{cmyk}{0,0,0,0.8}%
    \definecolor{TUMGray2}{cmyk}{0,0,0,0.5}%
    \definecolor{TUMGray3}{cmyk}{0,0,0,0.2}%
    % Accent Colors in CMYK
    \definecolor{TUMBlue3}{cmyk}{0.65,0.19,0.01,0.04}% Pantone 542
    \definecolor{TUMBlue4}{cmyk}{0.42,0.09,0,0}%       Pantone 283
    \definecolor{TUMIvory}{cmyk}{0.03,0.04,0.14,0.08}% Pantone 7527
    \definecolor{TUMOrange}{cmyk}{0,0.65,0.95,0}%	   Pantone 158
    \definecolor{TUMGreen}{cmyk}{0.35,0,1,0.2}%        Pantone 383
    % Extended Diagram Colors in CMYK
    \definecolor{TUMDiagPurple}   {cmyk}{ 0.5,   1,    0,  0.4}%
    \definecolor{TUMDiagBlue}     {cmyk}{   1,   1,    0,  0.4}%
    \definecolor{TUMDiagBlueGreen}{cmyk}{   1,0.03,  0.3,  0.3}%
    \definecolor{TUMDiagGreen1}   {cmyk}{   1,   0,    1,  0.2}%
    \definecolor{TUMDiagGreen2}   {cmyk}{ 0.6,   0,    1,  0.2}%
    \definecolor{TUMDiagYellow}   {cmyk}{   0, 0.1,    1,    0}%
    \definecolor{TUMDiagOrange1}  {cmyk}{   0, 0.3,    1,    0}%
    \definecolor{TUMDiagOrange2}  {cmyk}{   0, 0.8,    1,  0.1}%
    \definecolor{TUMDiagRed1}     {cmyk}{ 0.1,   1,    1,  0.1}%
    \definecolor{TUMDiagRed2}     {cmyk}{   0,   1     1,  0.4}%
    % Anniversary colors "150 Jahre culture of excellence" (for use in 2018 only)
    \definecolor{TUMAnniMustard}{cmyk}{12,18,85,0}%     Pantone 7405
    \definecolor{TUMAnniMustard2}{cmyk}{18,25,90,10}%   Pantone 103
    \definecolor{TUMAnniOrange}{cmyk}{0,40,100,0}%      Pantone 137
    \definecolor{TUMAnniOrange2}{cmyk}{0,50,100,0}%     Pantone 151
    \definecolor{TUMAnniGreen}{cmyk}{35,0,95,0}%        Pantone 382
    \definecolor{TUMAnniGreen2}{cmyk}{45,5,100,0}%      Pantone 376
    \definecolor{TUMAnniBlue}{cmyk}{45,0,3,0}%          Pantone 297
    \definecolor{TUMAnniBlue2}{cmyk}{60,0,0,0}%         Pantone 306
    \definecolor{TUMAnniPink}{cmyk}{0,55,30,0}%         Pantone 708
    \definecolor{TUMAnniPink2}{cmyk}{8,60,30,0}%        Pantone 702
\else%
    % Primary Colors in RGB
    \definecolor{TUMBlue}{RGB}{0,101,189}%      Pantone 300 C
    \definecolor{TUMWhite}{RGB}{255,255,255}%
    \definecolor{TUMBlack}{RGB}{0,0,0}%
    % Secondary Colors in RGB
    \definecolor{TUMBlue1}{RGB}{0,51,89}%       Pantone 540
    \definecolor{TUMBlue2}{RGB}{0,82,147}%      Pantone 301
    \definecolor{TUMGray1}{RGB}{51,51,51}%
    \definecolor{TUMGray2}{RGB}{127,127,127}%
    \definecolor{TUMGray3}{RGB}{204,204,204}%
    % Accent Colors in RGB
    \definecolor{TUMBlue3}{RGB}{100,160,200}%   Pantone 542
    \definecolor{TUMBlue4}{RGB}{152,198,234}%   Pantone 283
    \definecolor{TUMIvory}{RGB}{218,215,203}%   Pantone 7527
    \definecolor{TUMOrange}{RGB}{227,114,34}%	Pantone 158
    \definecolor{TUMGreen}{RGB}{162,173,0}%     Pantone 383
    % Extended Diagram Colors in RGB
    \definecolor{TUMDiagPurple}   {RGB}{105,  8, 90}%
    \definecolor{TUMDiagBlue}     {RGB}{ 15, 27, 95}%
    \definecolor{TUMDiagBlueGreen}{RGB}{  0,119,138}%
    \definecolor{TUMDiagGreen1}   {RGB}{  0,124, 48}%
    \definecolor{TUMDiagGreen2}   {RGB}{103,154, 29}%
    \definecolor{TUMDiagYellow}   {RGB}{255,255,  0}%
    \definecolor{TUMDiagOrange1}  {RGB}{249,186,  0}%
    \definecolor{TUMDiagOrange2}  {RGB}{214, 76, 19}%
    \definecolor{TUMDiagRed1}     {RGB}{196,  7, 27}%
    \definecolor{TUMDiagRed2}     {RGB}{156, 13, 22}%
    % Anniversary colors "150 Jahre culture of excellence" (for use in 2018 only)
    \definecolor{TUMAnniMustard}{RGB}{232,200,55}%  Pantone 7405
    \definecolor{TUMAnniMustard2}{RGB}{202,171,41}% Pantone 103
    \definecolor{TUMAnniOrange}{RGB}{247,166,0}%    Pantone 137
    \definecolor{TUMAnniOrange2}{RGB}{243,145,0}%   Pantone 151
    \definecolor{TUMAnniGreen}{RGB}{188,207,30}%    Pantone 382
    \definecolor{TUMAnniGreen2}{RGB}{162,191,22}%   Pantone 376
    \definecolor{TUMAnniBlue}{RGB}{146,212,241}%    Pantone 297
    \definecolor{TUMAnniBlue2}{RGB}{91,197,242}%    Pantone 306
    \definecolor{TUMAnniPink}{RGB}{242,144,149}%    Pantone 708
    \definecolor{TUMAnniPink2}{RGB}{227,130,143}%   Pantone 702
\fi%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMcolor.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%===================
%--- Definitions ---
%=================== 
% AM colors
\colorlet{AMLogoBlue1}{TUMBlue}%
\colorlet{AMLogoBlue2}{TUMBlue4}%
%
%
%
\else\fi% End-check if module AMcolor is enabled
%###################################################################################################
%--- END MODULE AMcolor ----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMtikz ---------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMtikz% Start-check if module AMtikz is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMtikz.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{pdfpages}%
\RequirePackage{tikz}%
\usetikzlibrary{calc}%
\usetikzlibrary{external}%
\ifdefined\optTikzExternalize%
    \tikzexternalize[prefix=figures/tikzoutput/,optimize command away=\includepdf]%
\else\fi%
\PassOptionsToPackage{framemethod=tikz}{mdframed}%
\RequirePackage{mdframed}%
\RequirePackage{pgfplots}%
\pgfplotsset{compat=newest}%        for Matlab2TikZ
\usetikzlibrary{plotmarks}%         for Matlab2TikZ
\usetikzlibrary{arrows.meta}%       for Matlab2TikZ
\usepgfplotslibrary{patchplots}%    for Matlab2TikZ
\RequirePackage{grffile}%           for Matlab2TikZ
\RequirePackage{pgfgantt}%
%
%
%===================
%--- Definitions ---
%===================
\pgfplotsset{plot coordinates/math parser=false}%   for Matlab2TikZ
\newlength\figureheight%                            for Matlab2TikZ
\newlength\figurewidth%                             for Matlab2TikZ
%
%
%================
%--- Commands ---
%================
% Misc
% ----
\newcommand{\AMtikzExternalizeSkipNext}{\tikzset{external/export next=false}}%
%
% Grids
% -----
\newcommand{\AMtikzGrid}[5][line width=1mm]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[inner sep=0pt, outer sep=0pt]%
        % Horizontal lines:
        \draw[#1] ($ (0mm,0mm) $) -- ($ (#2*#4,0mm) $);%
        \AMcoreRepetition{#5}{\draw[#1] ($ (0mm,#3*\Step) $) -- ($ (#2*#4,#3*\Step) $);}%
        % Vertical lines:
        \draw[#1] ($ (0mm,0mm) $) -- ($ (0mm,#3*#5) $);%
        \AMcoreRepetition{#4}{\draw[#1] ($ (#2*\Step,0mm) $) -- ($ (#2*\Step,#3*#5) $);}%
    \end{tikzpicture}%
}%
%
% Frames
% ------
\newcommand{\AMtikzCircled}[1]{%
    \AMtikzExternalizeSkipNext%
    \tikz[baseline=(char.base)]{\node[shape=circle,draw,inner sep=1pt] (char) {#1};}%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMtikz.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%================
%--- Commands ---
%================
% AM snake
% --------
\newcommand{\AMtikzAMsnake}[5]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[inner sep=0pt, outer sep=0pt]%
        \path[draw=#5,line join=round,line cap=round,line width=#4]%
            (0mm,0mm) -- ($ (#3,0mm) $) .. controls ($ (#3+#2/10*1,0mm) $) and ($ (#3+#2/10*2,0mm) $) .. ($ (#3+#2/10*3,#2/10) $) -- ($ (#3+#2/10*12,#2/10*9) $) .. controls ($ (#3+#2/10*13,#2) $) and ($ (#3+#2/10*14,#2) $) .. ($ (#3+#2/10*15,#2) $) -- ($ (#1,#2) $);%
    \end{tikzpicture}%
}%
%
%
%
\else\fi% End-check if module AMtikz is enabled
%###################################################################################################
%--- END MODULE AMtikz -----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMlogo ---------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMlogo% Start-check if module AMlogo is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMlogo.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Commands ---
%================
% Tercets
% -------
\newcommand{\AMlogoTercetTUM}[2][TUMBlue]{%
    \resizebox*{!}{#2}{%
        \color{#1}%
        \fontsize{8.5}{8.5}\fontfamily{phv}\selectfont%
        \begin{tikzpicture}%
            \node[anchor=south west] at (0cm,0.66cm) {\vphantom{Xg}};%
            \node[anchor=south west] at (0cm,0.33cm) {\vphantom{Xg}};%
            \node[anchor=south west] at (0cm,0cm) {\vphantom{Xg}\AMlangTUM};%
        \end{tikzpicture}%
    }%
}%
\newcommand{\AMlogoTercetSchool}[2][TUMBlue]{%
    \resizebox*{!}{#2}{%
        \color{#1}%
        \fontsize{8.5}{8.5}\fontfamily{phv}\selectfont%
        \begin{tikzpicture}%
            \node[anchor=south west] at (0cm,0.66cm) {\vphantom{Xg}};%
            \node[anchor=south west] at (0cm,0.33cm) {\vphantom{Xg}\AMlangSchool};%
            \node[anchor=south west] at (0cm,0cm) {\vphantom{Xg}\AMlangTUM};%
        \end{tikzpicture}%
    }%
}%
\newcommand{\AMlogoTercetChair}[2][TUMBlue]{%
    \resizebox*{!}{#2}{%
        \color{#1}%
        \fontsize{8.5}{8.5}\fontfamily{phv}\selectfont%
        \begin{tikzpicture}%
            \node[anchor=south west] at (0cm,0.66cm) {\vphantom{Xg}\AMlangChair};%
            \node[anchor=south west] at (0cm,0.33cm) {\vphantom{Xg}\AMlangSchool};%
            \node[anchor=south west] at (0cm,0cm) {\vphantom{Xg}\AMlangTUM};%
        \end{tikzpicture}%
    }%
}%
%
% Logos
% -----
\newcommand{\AMlogoTUM}[3][!]{%
    \resizebox{#1}{#2}{%
        \AMtikzExternalizeSkipNext%
        \begin{tikzpicture}[y=0.80pt, x=0.80pt, yscale=-1, xscale=1, inner sep=0pt, outer sep=0pt]%
        \begin{scope}[even odd rule,line width=0.800pt]%
          \begin{scope}[fill=#3]%
            \path[fill] (405,0) -- (370,0) -- (370,320) -- (290,320) -- (290,0) -- (255,0) -- (0,0) --
                (0,70) -- (70,70) -- (70,390) -- (140,390) -- (140,70) -- (220,70) -- (220,390) --
                (255,390) -- (405,390) -- (440,390) -- (440,70) -- (520,70) -- (520,390) -- (590,390) --
                (590,70) --  (670,70) -- (670,390) -- (740,390) -- (740,35) -- (740,0) -- (405,0) -- cycle;
          \end{scope}%
        \end{scope}%
        \end{tikzpicture}%
    }%
}%
\newcommand{\AMlogoDepartmentED}[3][!]{%
    \resizebox{#1}{#2}{%
        \AMtikzExternalizeSkipNext%
        \begin{tikzpicture}[y=0.80pt, x=0.80pt, yscale=-1, xscale=1, inner sep=0pt, outer sep=0pt]%
            \begin{scope}[even odd rule,line width=0.800pt]%
                \begin{scope}[shift={(-3366.0,-39.0)},fill=#3]%
                    \path[fill] (3848.2448,185.1588) .. controls (3842.0586,169.7349) and
                        (3827.6725,147.4471) .. (3816.1689,135.4551) .. controls (3814.8263,134.0613)
                        and (3783.0794,101.2148) .. (3746.9364,86.3300) -- (3743.6459,94.2852) ..
                        controls (3778.0910,108.4731) and (3809.6141,141.0829) .. (3809.9432,141.4117)
                        .. controls (3820.8019,152.7330) and (3834.3983,173.7980) ..
                        (3840.2291,188.3672) -- (3867.1192,255.3883) -- (3875.1217,252.1799) --
                        (3848.2448,185.1588) -- (3848.2448,185.1588) -- cycle(3475.7058,358.7799) ..
                        controls (3477.0352,360.1737) and (3508.8084,393.0071) .. (3544.9251,407.8787)
                        -- (3548.2024,399.9104) .. controls (3513.7705,385.7487) and
                        (3482.2342,353.1521) .. (3481.9314,352.8365) .. controls (3471.0596,341.4888)
                        and (3457.4763,320.4370) .. (3451.6324,305.8809) -- (3424.7291,238.8204) --
                        (3416.7266,242.0288) -- (3443.6298,309.0762) .. controls (3449.8160,324.5001)
                        and (3464.2022,346.7879) .. (3475.7058,358.7799) -- (3475.7058,358.7799) --
                        cycle(3464.7813,238.7152) .. controls (3462.5042,221.2663) and
                        (3462.4911,203.5413) .. (3464.7813,186.0266) .. controls (3467.2952,166.7895)
                        and (3470.4410,157.8612) .. (3470.4541,157.7692) .. controls
                        (3475.8111,142.9369) and (3488.0913,120.7938) .. (3497.8444,108.3942) --
                        (3542.7270,51.3796) -- (3535.9617,46.0673) -- (3491.0659,103.0819) .. controls
                        (3480.8522,116.0733) and (3467.9665,139.3078) .. (3462.3595,154.8501) ..
                        controls (3462.2147,155.2445) and (3458.8847,164.5936) .. (3456.2391,184.9090)
                        .. controls (3453.8568,203.1731) and (3453.8568,221.6476) ..
                        (3456.2523,239.8329) -- (3464.7813,238.7152) -- (3464.7813,238.7152) --
                        cycle(3636.4277,52.5498) .. controls (3651.3798,48.7366) and
                        (3676.9011,47.3691) .. (3692.1559,49.5650) -- (3764.0339,59.8476) --
                        (3765.2448,51.3270) -- (3693.3931,41.0312) .. controls (3676.9274,38.6907) and
                        (3650.4190,40.1108) .. (3634.2823,44.2001) .. controls (3632.3870,44.6998) and
                        (3587.4122,56.3236) .. (3557.4422,79.3346) -- (3562.6939,86.1590) .. controls
                        (3591.2292,64.2394) and (3635.9802,52.6682) .. (3636.4277,52.5498) --
                        (3636.4277,52.5498) -- cycle(3644.9173,441.2907) -- (3366.0000,441.2907) --
                        (3366.0000,449.9033) -- (3645.9965,449.9033) -- (3645.9965,449.8376) ..
                        controls (3753.8728,449.2459) and (3841.4795,361.4360) .. (3841.4795,253.5080)
                        -- (3832.8583,253.5080) .. controls (3832.8583,357.0442) and
                        (3748.5422,441.2907) .. (3644.9173,441.2907) -- (3644.9173,441.2907) -- cycle;
                \end{scope}%
            \end{scope}%
        \end{tikzpicture}%
    }%
}%
\newcommand{\AMlogoDepartmentIN}[3][!]{%
    \resizebox{#1}{#2}{%
        \AMtikzExternalizeSkipNext%
        \begin{tikzpicture}[y=0.80pt, x=0.80pt, yscale=-1, xscale=1, inner sep=0pt, outer sep=0pt]%
            \begin{scope}[even odd rule,line width=0.800pt]%
                \begin{scope}[shift={(-3928.0,0)},fill=#3]%
                \path[fill] (4128.8399,404.6691) -- (4137.1738,404.6691) -- (4137.1738,0.0000)
                -- (4128.8399,0.0000) -- (4128.8399,404.6691) -- cycle(4150.0535,41.1873) --
                (4149.3647,49.4799) .. controls (4250.4597,57.8138) and (4329.6523,143.8940)
                .. (4329.6523,245.4848) .. controls (4329.6523,353.8943) and
                (4241.4370,442.1096) .. (4132.9862,442.1096) .. controls (4024.5492,442.1096)
                and (3936.3201,353.8943) .. (3936.3201,245.4848) .. controls
                (3936.3201,143.8802) and (4015.5265,57.8000) .. (4116.6353,49.4799) --
                (4115.9603,41.1873) .. controls (4010.5538,49.8381) and (3928.0000,139.5962)
                .. (3928.0000,245.4848) .. controls (3928.0000,358.5227) and
                (4019.9621,450.4573) .. (4132.9862,450.4573) .. controls (4246.0379,450.4573)
                and (4338.0000,358.5227) .. (4338.0000,245.4848) .. controls
                (4338.0000,139.6099) and (4255.4462,49.8518) .. (4150.0535,41.1873) --
                (4150.0535,41.1873) -- cycle;
                \end{scope}%
            \end{scope}%
        \end{tikzpicture}%
    }%
}%
\newcommand{\AMlogoDepartmentCIT}[3][!]{%
    % CIT has no logo (yet). Provide empty command for compatability.
}%
\newcommand{\AMlogoDepartment}[3][!]{\AMlogoDepartmentED[#1]{#2}{#3}}%
%
% Badges
% ------
\newcommand{\AMlogoAnniversary}[2][1cm]{%
    \resizebox{!}{2.654928#1}{%
        \AMtikzExternalizeSkipNext%
        \begin{tikzpicture}[y=0.80pt, x=0.80pt, yscale=-1, xscale=1, inner sep=0pt, outer sep=0pt]%
            \begin{scope}[shift={(0,-0.00391)}]%
                \path[fill=#2,nonzero rule] (0.0000,0.0039) -- (12.4844,75.2617) --
                    (118.8750,56.6875) -- (118.8750,0.0039) -- (0.0000,0.0039);
                \path[fill=TUMWhite,nonzero rule] (22.6523,28.5781) .. controls (22.6133,28.5586)
                    and (22.2930,28.3984) .. (22.2539,28.3789) .. controls (21.6797,28.0938) and
                    (21.0977,27.9492) .. (20.5156,27.9492) .. controls (19.8047,27.9492) and
                    (19.1758,28.0938) .. (18.6484,28.3750) .. controls (18.1211,28.6523) and
                    (17.7266,29.0352) .. (17.4375,29.5430) .. controls (17.1523,30.0430) and
                    (17.0039,30.6172) .. (17.0039,31.3086) .. controls (17.0039,32.0000) and
                    (17.1523,32.5859) .. (17.4375,33.0898) .. controls (17.7266,33.5938) and
                    (18.1211,33.9766) .. (18.6484,34.2578) .. controls (19.1797,34.5391) and
                    (19.8047,34.6797) .. (20.5156,34.6797) .. controls (21.1406,34.6797) and
                    (21.7617,34.5156) .. (22.3750,34.1914) .. controls (22.3750,34.1914) and
                    (22.4297,34.1641) .. (22.4531,34.1445) .. controls (22.5039,34.1016) and
                    (22.5156,34.0391) .. (22.4844,33.9492) -- (22.1094,33.1836) .. controls
                    (22.0508,33.0781) and (21.9648,33.0391) .. (21.8555,33.0742) .. controls
                    (21.8555,33.0742) and (21.6563,33.1602) .. (21.5547,33.1914) .. controls
                    (20.8906,33.4063) and (20.0352,33.3203) .. (19.5469,33.0703) .. controls
                    (19.2461,32.9141) and (19.0039,32.6367) .. (18.8203,32.3438) .. controls
                    (18.6406,32.0508) and (18.5469,31.7031) .. (18.5469,31.3164) .. controls
                    (18.5469,30.9258) and (18.6406,30.5820) .. (18.8203,30.2891) .. controls
                    (19.0039,29.9961) and (19.2461,29.7148) .. (19.5469,29.5625) .. controls
                    (20.0352,29.3125) and (20.9414,29.2227) .. (21.6055,29.4375) .. controls
                    (21.6563,29.4570) and (21.7344,29.4844) .. (21.7969,29.5117) .. controls
                    (21.8594,29.5352) and (22.1563,29.6875) .. (22.1563,29.6875) .. controls
                    (22.2656,29.7227) and (22.3477,29.6836) .. (22.4063,29.5781) --
                    (22.7656,28.8242) .. controls (22.7969,28.7344) and (22.7852,28.6719) ..
                    (22.7305,28.6250) .. controls (22.7109,28.6094) and (22.6523,28.5781) ..
                    (22.6523,28.5781);
                \path[fill=TUMWhite,nonzero rule] (28.9766,28.1328) -- (27.6992,28.1328) ..
                    controls (27.5938,28.1328) and (27.5313,28.1758) .. (27.5117,28.2656) --
                    (27.5117,32.5078) -- (27.4023,32.6641) .. controls (27.1211,32.9609) and
                    (26.7305,33.1055) .. (26.4609,33.1094) .. controls (26.2305,33.1133) and
                    (26.1133,33.0937) .. (25.9648,33.0234) .. controls (25.5625,32.8438) and
                    (25.3789,32.5195) .. (25.3789,32.0430) -- (25.3789,28.2188) .. controls
                    (25.3633,28.1641) and (25.3086,28.1328) .. (25.2188,28.1328) --
                    (23.9453,28.1328) .. controls (23.8125,28.1328) and (23.7461,28.1992) ..
                    (23.7461,28.3281) -- (23.7461,32.2266) .. controls (23.7461,32.7500) and
                    (23.8398,33.2031) .. (24.0273,33.5742) .. controls (24.2031,33.9297) and
                    (24.4570,34.1992) .. (24.7969,34.3867) .. controls (25.1445,34.5820) and
                    (25.6406,34.6797) .. (25.8672,34.6797) .. controls (26.9414,34.6797) and
                    (27.2578,34.1992) .. (27.6836,33.6836) .. controls (27.6836,33.6836) and
                    (27.9141,34.3516) .. (27.9297,34.3750) .. controls (27.9766,34.4492) and
                    (28.0313,34.4844) .. (28.0938,34.4844) -- (28.9453,34.4844) .. controls
                    (28.9961,34.4844) and (29.0391,34.4648) .. (29.0781,34.4258) .. controls
                    (29.1133,34.3867) and (29.1367,34.3477) .. (29.1367,34.3047) .. controls
                    (29.1367,34.3047) and (29.1406,28.2891) .. (29.1406,28.2617) .. controls
                    (29.1406,28.1758) and (29.0859,28.1328) .. (28.9766,28.1328);
                \path[fill=TUMWhite,nonzero rule] (33.6758,20.3789) .. controls (33.6758,20.5039)
                    and (33.7344,20.5664) .. (33.8594,20.5664) -- (36.0156,20.5664) .. controls
                    (36.1328,20.5664) and (36.1914,20.5039) .. (36.1914,20.3789) --
                    (36.1914,12.1680) .. controls (36.1914,12.0625) and (36.1328,12.0234) ..
                    (36.0156,12.0430) -- (32.8711,12.5313) .. controls (32.7383,12.5469) and
                    (32.6719,12.6094) .. (32.6719,12.7148) -- (32.6719,14.2188) .. controls
                    (32.6562,14.3164) and (32.7109,14.3672) .. (32.8359,14.3672) --
                    (33.6758,14.3672) -- (33.6758,20.3789);
                \path[fill=TUMWhite,nonzero rule] (32.0430,25.8594) -- (30.7656,25.8594) ..
                    controls (30.6562,25.8594) and (30.5938,25.9062) .. (30.5742,25.9922) --
                    (30.5742,34.3555) .. controls (30.5742,34.4414) and (30.6289,34.4844) ..
                    (30.7383,34.4844) -- (32.0156,34.4844) .. controls (32.1250,34.4844) and
                    (32.1875,34.4414) .. (32.2070,34.3516) -- (32.2070,25.9883) .. controls
                    (32.2070,25.9023) and (32.1523,25.8594) .. (32.0430,25.8594);
                \path[fill=TUMWhite,nonzero rule] (65.4648,26.9688) .. controls (65.3008,27.2969)
                    and (65.2109,27.6875) .. (65.1914,28.1328) -- (64.3945,28.1328) .. controls
                    (64.3047,28.1289) and (64.2617,28.1719) .. (64.2617,28.2695) --
                    (64.2617,28.9883) .. controls (64.2617,29.0234) and (64.2773,29.0547) ..
                    (64.3047,29.0820) .. controls (64.3359,29.1055) and (64.3633,29.1172) ..
                    (64.3945,29.1172) -- (65.1836,29.1172) -- (65.1875,34.3438) .. controls
                    (65.1875,34.3828) and (65.1992,34.4180) .. (65.2305,34.4453) .. controls
                    (65.2578,34.4688) and (65.2891,34.4844) .. (65.3242,34.4844) --
                    (66.6797,34.4844) .. controls (66.7148,34.4844) and (66.7461,34.4687) ..
                    (66.7773,34.4414) .. controls (66.8047,34.4141) and (66.8203,34.3789) ..
                    (66.8203,34.3437) -- (66.8203,29.1172) -- (67.7187,29.1172) .. controls
                    (67.8008,29.1172) and (67.8477,29.0703) .. (67.8594,28.9805) --
                    (67.9570,28.2578) .. controls (67.9570,28.1758) and (67.9102,28.1328) ..
                    (67.8203,28.1328) -- (66.8203,28.1328) .. controls (66.8242,27.6953) and
                    (66.9023,27.4297) .. (67.0977,27.2266) .. controls (67.3008,27.0156) and
                    (67.4922,26.9453) .. (67.8438,26.9453) -- (67.8711,26.9453) .. controls
                    (67.9102,26.9453) and (67.9414,26.9336) .. (67.9688,26.9023) .. controls
                    (67.9961,26.8750) and (68.0117,26.8437) .. (68.0117,26.8086) --
                    (68.0117,26.0000) .. controls (68.0117,25.9648) and (67.9961,25.9297) ..
                    (67.9688,25.9023) .. controls (67.9375,25.8750) and (67.9063,25.8594) ..
                    (67.8711,25.8594) -- (67.5391,25.8594) -- (67.5391,25.8633) .. controls
                    (67.5273,25.8633) and (67.5195,25.8594) .. (67.5078,25.8594) .. controls
                    (67.0117,25.8594) and (66.5859,25.9609) .. (66.2383,26.1523) .. controls
                    (65.8984,26.3437) and (65.6445,26.6133) .. (65.4648,26.9687);
                \path[fill=TUMWhite,nonzero rule] (47.1328,28.0234) .. controls (46.4883,28.1797)
                    and (46.2695,28.3555) .. (45.9453,28.7500) .. controls (45.9453,28.7500) and
                    (45.7773,28.2656) .. (45.7617,28.2422) .. controls (45.7148,28.1680) and
                    (45.6562,28.1328) .. (45.5938,28.1328) -- (44.8477,28.1328) .. controls
                    (44.7969,28.1328) and (44.7539,28.1523) .. (44.7148,28.1914) .. controls
                    (44.6758,28.2266) and (44.6563,28.2695) .. (44.6563,28.3086) .. controls
                    (44.6563,28.3086) and (44.6485,34.3281) .. (44.6485,34.3555) .. controls
                    (44.6485,34.4414) and (44.7031,34.4844) .. (44.8125,34.4844) --
                    (46.0938,34.4844) .. controls (46.1992,34.4844) and (46.2617,34.4414) ..
                    (46.2813,34.3516) -- (46.2813,29.7031) .. controls (46.4219,29.4336) and
                    (46.7891,29.2578) .. (47.1836,29.2578) -- (47.7149,29.2578) .. controls
                    (47.7969,29.2578) and (47.8399,29.2070) .. (47.8399,29.0977) --
                    (47.8399,28.1797) .. controls (47.8399,28.1016) and (47.8164,28.0469) ..
                    (47.7695,28.0156) .. controls (47.7539,28.0039) and (47.7344,27.9961) ..
                    (47.7110,27.9922) .. controls (47.4961,27.9961) and (47.3203,27.9766) ..
                    (47.1328,28.0234);
                \path[fill=TUMWhite,nonzero rule] (63.3555,18.1797) .. controls (63.4492,18.1016)
                    and (63.5938,18.0625) .. (63.7891,18.0625) .. controls (63.9883,18.0625) and
                    (64.1914,18.1641) .. (64.3945,18.3672) -- (64.3945,18.6328) .. controls
                    (64.1914,18.8359) and (63.9844,18.9375) .. (63.7734,18.9375) .. controls
                    (63.5859,18.9375) and (63.4453,18.8984) .. (63.3516,18.8203) .. controls
                    (63.2578,18.7461) and (63.2148,18.6367) .. (63.2148,18.5039) .. controls
                    (63.2148,18.3672) and (63.2617,18.2617) .. (63.3555,18.1797) --
                    cycle(61.5938,20.4766) .. controls (61.9453,20.6602) and (62.3477,20.7500) ..
                    (62.8086,20.7500) .. controls (63.7656,20.7500) and (64.4648,20.5156) ..
                    (64.9062,20.0508) -- (64.9961,20.3789) .. controls (64.9961,20.4297) and
                    (65.0156,20.4687) .. (65.0547,20.5078) .. controls (65.0937,20.5469) and
                    (65.1367,20.5664) .. (65.1836,20.5664) -- (66.7226,20.5664) .. controls
                    (66.7734,20.5664) and (66.8164,20.5469) .. (66.8555,20.5078) .. controls
                    (66.8906,20.4688) and (66.9062,20.4297) .. (66.9062,20.3789) --
                    (66.9062,17.2227) .. controls (66.9062,16.0117) and (66.6601,15.1172) ..
                    (66.1601,14.5430) .. controls (65.6601,13.9687) and (64.8867,13.6836) ..
                    (63.8398,13.6836) .. controls (62.7930,13.6836) and (61.9062,13.9141) ..
                    (61.1758,14.3789) .. controls (61.1172,14.4219) and (61.0898,14.4727) ..
                    (61.0898,14.5273) .. controls (61.0898,14.5508) and (61.0976,14.5781) ..
                    (61.1094,14.6133) -- (61.8047,16.0703) .. controls (61.8398,16.1367) and
                    (61.8828,16.1680) .. (61.9453,16.1680) .. controls (61.9687,16.1680) and
                    (62.0000,16.1563) .. (62.0312,16.1367) .. controls (62.2539,16.0117) and
                    (62.4805,15.9180) .. (62.7109,15.8633) .. controls (62.9414,15.8086) and
                    (63.1914,15.7813) .. (63.4570,15.7813) .. controls (63.7383,15.7813) and
                    (63.9648,15.8594) .. (64.1367,16.0156) .. controls (64.3047,16.1758) and
                    (64.3945,16.3867) .. (64.3945,16.6484) -- (64.3945,17.1641) .. controls
                    (64.2734,16.9648) and (64.0703,16.8047) .. (63.7812,16.6875) .. controls
                    (63.4922,16.5703) and (63.1680,16.5117) .. (62.8086,16.5117) .. controls
                    (62.3438,16.5117) and (61.9336,16.5977) .. (61.5859,16.7656) .. controls
                    (61.2344,16.9336) and (60.9648,17.1758) .. (60.7734,17.4922) .. controls
                    (60.5859,17.8047) and (60.4883,18.1719) .. (60.4883,18.5977) .. controls
                    (60.4883,19.0117) and (60.5859,19.3828) .. (60.7773,19.7109) .. controls
                    (60.9688,20.0391) and (61.2422,20.2930) .. (61.5938,20.4766);
                \path[fill=TUMWhite,nonzero rule] (36.6680,28.1328) -- (35.6680,28.1328) --
                    (35.6680,26.4688) .. controls (35.6680,26.4297) and (35.6562,26.3984) ..
                    (35.6250,26.3711) .. controls (35.5977,26.3438) and (35.5664,26.3320) ..
                    (35.5312,26.3398) -- (34.1758,26.4687) .. controls (34.0820,26.4765) and
                    (34.0352,26.5195) .. (34.0352,26.5937) -- (34.0352,28.1328) --
                    (33.2422,28.1328) .. controls (33.1562,28.1289) and (33.1133,28.1719) ..
                    (33.1133,28.2695) -- (33.1133,28.9609) .. controls (33.1133,28.9961) and
                    (33.1289,29.0273) .. (33.1562,29.0547) .. controls (33.1836,29.0781) and
                    (33.2148,29.0937) .. (33.2422,29.0937) -- (34.0352,29.0937) --
                    (34.0352,34.3437) .. controls (34.0352,34.3828) and (34.0508,34.4180) ..
                    (34.0781,34.4453) .. controls (34.1094,34.4687) and (34.1406,34.4844) ..
                    (34.1758,34.4844) -- (35.5313,34.4844) .. controls (35.5664,34.4844) and
                    (35.5977,34.4687) .. (35.6250,34.4414) .. controls (35.6563,34.4141) and
                    (35.6680,34.3789) .. (35.6680,34.3437) -- (35.6680,29.0937) --
                    (36.5703,29.0937) .. controls (36.6484,29.0937) and (36.6953,29.0469) ..
                    (36.7109,28.9531) -- (36.8047,28.2578) .. controls (36.8047,28.1758) and
                    (36.7617,28.1328) .. (36.6680,28.1328);
                \path[fill=TUMWhite,nonzero rule] (62.2109,32.2930) .. controls (62.0469,32.6016)
                    and (61.8164,32.8438) .. (61.5234,33.0156) .. controls (61.2266,33.1875) and
                    (60.8945,33.2734) .. (60.5273,33.2734) .. controls (60.1602,33.2734) and
                    (59.8281,33.1875) .. (59.5352,33.0195) .. controls (59.2383,32.8516) and
                    (59.0078,32.6133) .. (58.8438,32.3047) .. controls (58.6914,32.0117) and
                    (58.6133,31.6797) .. (58.6133,31.3125) .. controls (58.6133,30.9453) and
                    (58.6914,30.6133) .. (58.8438,30.3203) .. controls (59.0078,30.0117) and
                    (59.2383,29.7734) .. (59.5352,29.6016) .. controls (59.8281,29.4336) and
                    (60.1602,29.3516) .. (60.5273,29.3516) .. controls (61.0859,29.3516) and
                    (61.5547,29.5352) .. (61.9141,29.9023) .. controls (62.2656,30.2695) and
                    (62.4375,30.7383) .. (62.4375,31.2969) .. controls (62.4375,31.6641) and
                    (62.3594,32.0000) .. (62.2109,32.2930) -- cycle(62.3359,28.3672) .. controls
                    (61.8164,28.0859) and (61.2109,27.9453) .. (60.5273,27.9453) .. controls
                    (59.8477,27.9453) and (59.2422,28.0859) .. (58.7266,28.3672) .. controls
                    (58.2109,28.6445) and (57.8203,29.0273) .. (57.5312,29.5352) .. controls
                    (57.2461,30.0430) and (57.1016,30.6406) .. (57.1016,31.3203) .. controls
                    (57.1016,31.9922) and (57.2461,32.5859) .. (57.5312,33.0898) .. controls
                    (57.8164,33.5977) and (58.2031,33.9805) .. (58.7188,34.2578) .. controls
                    (59.2344,34.5391) and (59.8438,34.6797) .. (60.5273,34.6797) .. controls
                    (61.2109,34.6797) and (61.8164,34.5391) .. (62.3320,34.2539) .. controls
                    (62.8477,33.9727) and (63.2383,33.5898) .. (63.5234,33.0820) .. controls
                    (63.8047,32.5781) and (63.9531,31.9766) .. (63.9531,31.3047) .. controls
                    (63.9531,30.6328) and (63.8086,30.0391) .. (63.5234,29.5352) .. controls
                    (63.2383,29.0273) and (62.8477,28.6445) .. (62.3359,28.3672);
                \path[fill=TUMWhite,nonzero rule] (47.3750,15.2031) .. controls (47.4883,14.8711)
                    and (47.6484,14.6172) .. (47.8555,14.4336) .. controls (48.0625,14.2500) and
                    (48.3008,14.1602) .. (48.5781,14.1602) .. controls (48.8477,14.1602) and
                    (49.0859,14.2500) .. (49.2930,14.4297) .. controls (49.5039,14.6094) and
                    (49.6641,14.8633) .. (49.7812,15.1914) .. controls (49.8945,15.5195) and
                    (49.9531,15.8984) .. (49.9531,16.3281) .. controls (49.9531,16.7656) and
                    (49.8945,17.1523) .. (49.7812,17.4883) .. controls (49.6641,17.8242) and
                    (49.5039,18.0820) .. (49.2930,18.2695) .. controls (49.0859,18.4570) and
                    (48.8477,18.5508) .. (48.5781,18.5508) .. controls (48.3125,18.5508) and
                    (48.0742,18.4570) .. (47.8633,18.2734) .. controls (47.6562,18.0898) and
                    (47.4961,17.8320) .. (47.3789,17.5000) .. controls (47.2656,17.1680) and
                    (47.2070,16.7852) .. (47.2070,16.3516) .. controls (47.2070,15.9141) and
                    (47.2617,15.5313) .. (47.3750,15.2031) -- cycle(46.5078,20.2031) .. controls
                    (47.1094,20.5664) and (47.8008,20.7500) .. (48.5898,20.7500) .. controls
                    (49.3750,20.7500) and (50.0664,20.5664) .. (50.6641,20.2031) .. controls
                    (51.2578,19.8359) and (51.7188,19.3203) .. (52.0469,18.6562) .. controls
                    (52.3750,17.9883) and (52.5391,17.2227) .. (52.5391,16.3516) .. controls
                    (52.5391,15.4766) and (52.3750,14.7109) .. (52.0430,14.0469) .. controls
                    (51.7148,13.3828) and (51.2539,12.8672) .. (50.6562,12.5039) .. controls
                    (50.0625,12.1406) and (49.3711,11.9609) .. (48.5820,11.9609) .. controls
                    (47.7969,11.9609) and (47.1055,12.1406) .. (46.5078,12.5039) .. controls
                    (45.9062,12.8672) and (45.4453,13.3828) .. (45.1133,14.0469) .. controls
                    (44.7852,14.7109) and (44.6211,15.4766) .. (44.6211,16.3516) .. controls
                    (44.6211,17.2227) and (44.7852,17.9883) .. (45.1133,18.6562) .. controls
                    (45.4453,19.3203) and (45.9102,19.8359) .. (46.5078,20.2031);
                \path[fill=TUMWhite,nonzero rule] (52.7812,30.8242) -- (49.9219,30.8242) ..
                    controls (49.9219,30.0156) and (50.5898,29.5039) .. (51.4297,29.5039) ..
                    controls (52.2656,29.5039) and (52.7812,30.0156) .. (52.7812,30.8242) --
                    cycle(48.6289,29.5547) .. controls (48.3203,30.0586) and (48.1680,30.7344) ..
                    (48.1680,31.3164) .. controls (48.1680,31.8984) and (48.3125,32.6094) ..
                    (48.6406,33.0977) .. controls (49.2891,34.0703) and (50.2773,34.6758) ..
                    (51.5430,34.6797) .. controls (52.4258,34.6836) and (53.5547,34.1328) ..
                    (53.5508,34.1289) -- (53.5547,34.1328) .. controls (53.6680,34.0820) and
                    (53.7070,33.9961) .. (53.6719,33.8750) -- (53.3398,33.0234) .. controls
                    (53.2812,32.9062) and (53.1875,32.8750) .. (53.0703,32.9297) .. controls
                    (53.0703,32.9297) and (53.0664,32.9297) .. (53.0625,32.9297) .. controls
                    (52.4102,33.1289) and (52.2461,33.2695) .. (51.5234,33.2695) .. controls
                    (50.6367,33.2695) and (49.9219,32.6406) .. (49.9219,31.9258) --
                    (53.9024,31.9258) .. controls (54.0195,31.9102) and (54.0742,31.8828) ..
                    (54.1094,31.7188) .. controls (54.1719,31.4180) and (54.1680,31.0703) ..
                    (54.1680,30.8750) .. controls (54.1680,30.1953) and (54.0078,29.6953) ..
                    (53.7422,29.2500) .. controls (53.1250,28.2227) and (51.9922,27.9609) ..
                    (51.4492,27.9609) .. controls (50.9063,27.9609) and (49.5039,28.1211) ..
                    (48.6289,29.5547);
                \path[fill=TUMWhite,nonzero rule] (56.4414,41.9805) .. controls (56.4727,41.8867)
                    and (56.4141,41.8203) .. (56.3398,41.8203) -- (54.5586,41.8203) .. controls
                    (54.4727,41.8203) and (54.3984,41.8711) .. (54.3516,41.9688) --
                    (53.4453,43.9609) -- (52.5352,41.9688) .. controls (52.4922,41.8711) and
                    (52.4141,41.8203) .. (52.3281,41.8203) -- (50.5469,41.8203) .. controls
                    (50.4727,41.8203) and (50.4141,41.8867) .. (50.4453,41.9805) --
                    (52.3789,45.0195) -- (50.3867,48.1484) .. controls (50.3359,48.2461) and
                    (50.4062,48.3164) .. (50.4961,48.3164) -- (52.2656,48.3164) .. controls
                    (52.3867,48.3164) and (52.4570,48.2617) .. (52.4805,48.1602) --
                    (53.4375,45.9727) -- (54.4102,48.1602) .. controls (54.4297,48.2617) and
                    (54.5000,48.3164) .. (54.6211,48.3164) -- (56.3906,48.3164) .. controls
                    (56.4844,48.3164) and (56.5508,48.2461) .. (56.5000,48.1484) --
                    (54.5078,45.0195) -- (56.4414,41.9805);
                \path[fill=TUMWhite,nonzero rule] (83.0273,15.8828) .. controls (83.1914,15.7227)
                    and (83.4102,15.6406) .. (83.6758,15.6406) .. controls (83.9336,15.6406) and
                    (84.1367,15.7227) .. (84.2891,15.8828) .. controls (84.4375,16.0430) and
                    (84.5156,16.2617) .. (84.5156,16.5312) -- (82.7812,16.5312) .. controls
                    (82.7812,16.2617) and (82.8633,16.0430) .. (83.0273,15.8828) --
                    cycle(86.3086,20.0625) .. controls (86.3867,20.0078) and (86.4141,19.9492) ..
                    (86.3906,19.8789) -- (85.8672,18.3672) .. controls (85.8437,18.2891) and
                    (85.8086,18.2539) .. (85.7617,18.2539) .. controls (85.7383,18.2539) and
                    (85.7148,18.2578) .. (85.6875,18.2695) .. controls (85.1055,18.5430) and
                    (84.4805,18.6758) .. (83.8164,18.6758) .. controls (83.5117,18.6758) and
                    (83.2656,18.5938) .. (83.0742,18.4297) .. controls (82.8867,18.2617) and
                    (82.7930,18.0469) .. (82.7930,17.7813) -- (86.5937,17.7813) .. controls
                    (86.7109,17.7813) and (86.7734,17.7266) .. (86.7852,17.6172) .. controls
                    (86.8125,17.3672) and (86.8281,17.1211) .. (86.8281,16.8711) .. controls
                    (86.8281,16.1641) and (86.6953,15.5703) .. (86.4336,15.0938) .. controls
                    (86.1719,14.6172) and (85.8047,14.2617) .. (85.3359,14.0313) .. controls
                    (84.8633,13.7969) and (84.3164,13.6836) .. (83.6914,13.6836) .. controls
                    (83.0078,13.6836) and (82.4023,13.8281) .. (81.8672,14.1250) .. controls
                    (81.3359,14.4219) and (80.9219,14.8359) .. (80.6289,15.3750) .. controls
                    (80.3320,15.9102) and (80.1836,16.5273) .. (80.1836,17.2227) .. controls
                    (80.1836,17.9180) and (80.3320,18.5313) .. (80.6328,19.0664) .. controls
                    (80.9297,19.6016) and (81.3477,20.0156) .. (81.8828,20.3086) .. controls
                    (82.4180,20.6016) and (83.0273,20.7500) .. (83.7187,20.7500) .. controls
                    (84.7578,20.7500) and (85.6211,20.5195) .. (86.3086,20.0625);
                \path[fill=TUMWhite,nonzero rule] (77.5898,44.6562) .. controls (77.5898,43.8477)
                    and (78.2578,43.3359) .. (79.1016,43.3359) .. controls (79.9336,43.3359) and
                    (80.4531,43.8477) .. (80.4531,44.6562) -- cycle(79.1172,41.7930) .. controls
                    (78.5742,41.7930) and (77.1719,41.9531) .. (76.2969,43.3867) .. controls
                    (75.9922,43.8906) and (75.8398,44.5703) .. (75.8398,45.1484) .. controls
                    (75.8398,45.7305) and (75.9805,46.4414) .. (76.3086,46.9297) .. controls
                    (76.9609,47.9023) and (77.9453,48.5078) .. (79.2109,48.5117) .. controls
                    (80.0977,48.5156) and (81.2227,47.9648) .. (81.2227,47.9609) --
                    (81.2227,47.9648) .. controls (81.3359,47.9141) and (81.3750,47.8281) ..
                    (81.3398,47.7070) -- (81.0117,46.8555) .. controls (80.9492,46.7383) and
                    (80.8594,46.7070) .. (80.7383,46.7617) .. controls (80.7383,46.7617) and
                    (80.7383,46.7617) .. (80.7305,46.7617) .. controls (80.0781,46.9609) and
                    (79.9180,47.1016) .. (79.1914,47.1016) .. controls (78.3047,47.1016) and
                    (77.5899,46.4727) .. (77.5899,45.7578) -- (81.5703,45.7578) .. controls
                    (81.6875,45.7422) and (81.7422,45.7148) .. (81.7774,45.5508) .. controls
                    (81.8438,45.2500) and (81.8399,44.8984) .. (81.8399,44.7070) .. controls
                    (81.8399,44.0273) and (81.6758,43.5273) .. (81.4102,43.0820) .. controls
                    (80.7930,42.0547) and (79.6602,41.7930) .. (79.1172,41.7930);
                \path[fill=TUMWhite,nonzero rule] (74.5469,39.6914) -- (73.2734,39.6914) ..
                    controls (73.1641,39.6914) and (73.1016,39.7383) .. (73.0820,39.8242) --
                    (73.0820,48.1875) .. controls (73.0820,48.2734) and (73.1367,48.3164) ..
                    (73.2461,48.3164) -- (74.5234,48.3164) .. controls (74.6328,48.3164) and
                    (74.6953,48.2734) .. (74.7148,48.1836) -- (74.7148,39.8203) .. controls
                    (74.7148,39.7344) and (74.6602,39.6914) .. (74.5469,39.6914);
                \path[fill=TUMWhite,nonzero rule] (75.9805,20.5664) -- (78.1328,20.5664) ..
                    controls (78.1797,20.5664) and (78.2227,20.5469) .. (78.2617,20.5078) ..
                    controls (78.2969,20.4688) and (78.3164,20.4297) .. (78.3164,20.3789) --
                    (78.3164,16.2930) .. controls (78.3984,16.1367) and (78.5313,16.0117) ..
                    (78.7266,15.9180) .. controls (78.9180,15.8281) and (79.1289,15.7813) ..
                    (79.3555,15.7813) -- (79.6719,15.7813) .. controls (79.7852,15.7891) and
                    (79.8438,15.7305) .. (79.8438,15.6133) -- (79.8438,14.0195) .. controls
                    (79.8438,13.9102) and (79.7852,13.8555) .. (79.6719,13.8555) --
                    (79.3555,13.8555) .. controls (78.6797,13.8555) and (78.1406,14.0977) ..
                    (77.7383,14.5820) -- (77.5586,14.0352) .. controls (77.5078,13.9141) and
                    (77.4453,13.8555) .. (77.3633,13.8555) -- (75.9805,13.8555) .. controls
                    (75.9336,13.8555) and (75.8906,13.8711) .. (75.8555,13.9102) .. controls
                    (75.8203,13.9453) and (75.8008,13.9805) .. (75.8008,14.0195) --
                    (75.8008,20.3789) .. controls (75.8008,20.4336) and (75.8203,20.4766) ..
                    (75.8555,20.5117) .. controls (75.8906,20.5469) and (75.9336,20.5664) ..
                    (75.9805,20.5664);
                \path[fill=TUMWhite,nonzero rule] (97.2656,44.6562) .. controls (97.2656,43.8477)
                    and (97.9297,43.3359) .. (98.7734,43.3359) .. controls (99.6094,43.3359) and
                    (100.1250,43.8477) .. (100.1250,44.6562) -- cycle(101.0859,43.0820) ..
                    controls (100.4648,42.0547) and (99.3359,41.7930) .. (98.7930,41.7930) ..
                    controls (98.2500,41.7930) and (96.8477,41.9531) .. (95.9688,43.3867) ..
                    controls (95.6641,43.8906) and (95.5117,44.5703) .. (95.5117,45.1484) ..
                    controls (95.5117,45.7305) and (95.6562,46.4414) .. (95.9805,46.9297) ..
                    controls (96.6328,47.9023) and (97.6211,48.5078) .. (98.8867,48.5117) ..
                    controls (99.7695,48.5156) and (100.8945,47.9648) .. (100.8945,47.9609) --
                    (100.8945,47.9648) .. controls (101.0117,47.9141) and (101.0508,47.8281) ..
                    (101.0117,47.7070) -- (100.6836,46.8555) .. controls (100.6211,46.7383) and
                    (100.5312,46.7070) .. (100.4141,46.7617) .. controls (100.4141,46.7617) and
                    (100.4101,46.7617) .. (100.4061,46.7617) .. controls (99.7537,46.9609) and
                    (99.5897,47.1016) .. (98.8670,47.1016) .. controls (97.9764,47.1016) and
                    (97.2654,46.4727) .. (97.2654,45.7578) -- (101.2420,45.7578) .. controls
                    (101.3592,45.7422) and (101.4139,45.7148) .. (101.4490,45.5508) .. controls
                    (101.5154,45.2500) and (101.5115,44.8984) .. (101.5115,44.7070) .. controls
                    (101.5115,44.0273) and (101.3514,43.5273) .. (101.0857,43.0820);
                \path[fill=TUMWhite,nonzero rule] (87.2539,42.0625) .. controls (86.9062,41.8672)
                    and (86.4102,41.7695) .. (86.1836,41.7695) .. controls (85.1094,41.7695) and
                    (84.7930,42.2500) .. (84.3672,42.7656) .. controls (84.3672,42.7656) and
                    (84.1367,42.0977) .. (84.1211,42.0742) .. controls (84.0742,42.0000) and
                    (84.0195,41.9648) .. (83.9570,41.9648) -- (83.0820,41.9648) .. controls
                    (83.0313,41.9648) and (82.9883,41.9844) .. (82.9492,42.0234) .. controls
                    (82.9102,42.0586) and (82.8906,42.1016) .. (82.8906,42.1406) .. controls
                    (82.8906,42.1406) and (82.9102,48.1602) .. (82.9102,48.1875) .. controls
                    (82.9102,48.2734) and (82.9648,48.3164) .. (83.0742,48.3164) --
                    (84.3516,48.3164) .. controls (84.4570,48.3164) and (84.5195,48.2734) ..
                    (84.5391,48.1836) -- (84.5391,43.9414) -- (84.6484,43.7852) .. controls
                    (84.9297,43.4883) and (85.3203,43.3437) .. (85.5898,43.3398) .. controls
                    (85.8203,43.3359) and (85.9336,43.3555) .. (86.0898,43.4258) .. controls
                    (86.4883,43.6055) and (86.6719,43.9297) .. (86.6719,44.4062) --
                    (86.6719,48.2305) .. controls (86.6875,48.2891) and (86.7422,48.3164) ..
                    (86.8281,48.3164) -- (88.1055,48.3164) .. controls (88.2383,48.3164) and
                    (88.3008,48.2500) .. (88.3008,48.1211) -- (88.3008,44.2227) .. controls
                    (88.3008,43.6992) and (88.2109,43.2461) .. (88.0234,42.8789) .. controls
                    (87.8438,42.5195) and (87.5938,42.2539) .. (87.2539,42.0625);
                \path[fill=TUMWhite,nonzero rule] (94.9883,42.4102) .. controls (94.9453,42.3906)
                    and (94.6289,42.2305) .. (94.5859,42.2109) .. controls (94.0117,41.9258) and
                    (93.4297,41.7812) .. (92.8477,41.7812) .. controls (92.1406,41.7812) and
                    (91.5117,41.9258) .. (90.9805,42.2070) .. controls (90.4570,42.4844) and
                    (90.0586,42.8672) .. (89.7695,43.3750) .. controls (89.4844,43.8750) and
                    (89.3359,44.4492) .. (89.3359,45.1406) .. controls (89.3359,45.8320) and
                    (89.4844,46.4180) .. (89.7695,46.9219) .. controls (90.0586,47.4258) and
                    (90.4570,47.8086) .. (90.9805,48.0898) .. controls (91.5117,48.3711) and
                    (92.1406,48.5117) .. (92.8477,48.5117) .. controls (93.4727,48.5117) and
                    (94.0938,48.3477) .. (94.7070,48.0234) .. controls (94.7070,48.0234) and
                    (94.7656,47.9961) .. (94.7852,47.9766) .. controls (94.8399,47.9336) and
                    (94.8477,47.8711) .. (94.8203,47.7812) -- (94.4414,47.0156) .. controls
                    (94.3828,46.9102) and (94.2969,46.8711) .. (94.1914,46.9062) .. controls
                    (94.1875,46.9062) and (93.9883,46.9922) .. (93.8867,47.0234) .. controls
                    (93.2227,47.2383) and (92.3672,47.1523) .. (91.8828,46.9023) .. controls
                    (91.5781,46.7461) and (91.3359,46.4687) .. (91.1524,46.1758) .. controls
                    (90.9727,45.8828) and (90.8828,45.5352) .. (90.8828,45.1445) .. controls
                    (90.8828,44.7578) and (90.9727,44.4141) .. (91.1524,44.1211) .. controls
                    (91.3359,43.8281) and (91.5781,43.5469) .. (91.8828,43.3945) .. controls
                    (92.3672,43.1445) and (93.2734,43.0547) .. (93.9375,43.2695) .. controls
                    (93.9922,43.2891) and (94.0664,43.3164) .. (94.1289,43.3437) .. controls
                    (94.1914,43.3672) and (94.4883,43.5195) .. (94.4883,43.5195) .. controls
                    (94.5977,43.5547) and (94.6797,43.5156) .. (94.7383,43.4102) --
                    (95.0977,42.6562) .. controls (95.1289,42.5664) and (95.1172,42.5039) ..
                    (95.0664,42.4570) .. controls (95.0430,42.4414) and (94.9883,42.4102) ..
                    (94.9883,42.4102);
                \path[fill=TUMWhite,nonzero rule] (59.2188,19.4609) .. controls (59.4492,19.0391)
                    and (59.5664,18.5391) .. (59.5664,17.9531) -- (59.5664,12.3281) .. controls
                    (59.5703,12.3125) and (59.5703,12.2930) .. (59.5703,12.2617) .. controls
                    (59.5703,12.1836) and (59.5195,12.1445) .. (59.4180,12.1445) --
                    (57.2695,12.1445) .. controls (57.1484,12.1445) and (57.0859,12.2070) ..
                    (57.0859,12.3281) -- (57.0859,17.6992) .. controls (57.0625,17.9492) and
                    (56.9727,18.1445) .. (56.8203,18.2891) .. controls (56.6680,18.4297) and
                    (56.4531,18.5039) .. (56.1758,18.5039) .. controls (55.8906,18.5039) and
                    (55.6133,18.4023) .. (55.3516,18.1992) .. controls (55.3047,18.1641) and
                    (55.2617,18.1445) .. (55.2266,18.1445) .. controls (55.1797,18.1445) and
                    (55.1445,18.1797) .. (55.1211,18.2461) -- (54.5234,19.9141) .. controls
                    (54.5195,19.9258) and (54.5195,19.9453) .. (54.5195,19.9687) .. controls
                    (54.5195,20.0234) and (54.5469,20.0820) .. (54.6094,20.1367) .. controls
                    (54.8359,20.3203) and (55.1328,20.4648) .. (55.5078,20.5781) .. controls
                    (55.8828,20.6953) and (56.2617,20.7500) .. (56.6445,20.7500) .. controls
                    (57.2578,20.7500) and (57.7812,20.6406) .. (58.2187,20.4180) .. controls
                    (58.6562,20.1992) and (58.9883,19.8789) .. (59.2187,19.4609);
                \path[fill=TUMWhite,nonzero rule] (45.6523,44.6562) .. controls (45.6523,43.8477)
                    and (46.3203,43.3359) .. (47.1641,43.3359) .. controls (47.9961,43.3359) and
                    (48.5156,43.8477) .. (48.5156,44.6562) -- cycle(47.1797,41.7930) .. controls
                    (46.6406,41.7930) and (45.2383,41.9531) .. (44.3594,43.3867) .. controls
                    (44.0547,43.8906) and (43.9023,44.5703) .. (43.9023,45.1484) .. controls
                    (43.9023,45.7305) and (44.0430,46.4414) .. (44.3711,46.9297) .. controls
                    (45.0234,47.9023) and (46.0117,48.5078) .. (47.2734,48.5117) .. controls
                    (48.1602,48.5156) and (49.2852,47.9648) .. (49.2852,47.9609) --
                    (49.2852,47.9648) .. controls (49.3984,47.9141) and (49.4414,47.8281) ..
                    (49.4023,47.7070) -- (49.0742,46.8555) .. controls (49.0117,46.7383) and
                    (48.9219,46.7070) .. (48.8008,46.7617) .. controls (48.8008,46.7617) and
                    (48.8008,46.7617) .. (48.7969,46.7617) .. controls (48.1445,46.9609) and
                    (47.9805,47.1016) .. (47.2578,47.1016) .. controls (46.3672,47.1016) and
                    (45.6523,46.4727) .. (45.6523,45.7578) -- (49.6328,45.7578) .. controls
                    (49.7500,45.7422) and (49.8047,45.7148) .. (49.8398,45.5508) .. controls
                    (49.9063,45.2500) and (49.9023,44.8984) .. (49.9023,44.7070) .. controls
                    (49.9023,44.0273) and (49.7422,43.5273) .. (49.4727,43.0820) .. controls
                    (48.8555,42.0547) and (47.7227,41.7930) .. (47.1797,41.7930);
                \path[fill=TUMWhite,nonzero rule] (43.0352,28.1328) -- (41.7578,28.1328) ..
                    controls (41.6523,28.1328) and (41.5898,28.1758) .. (41.5703,28.2656) --
                    (41.5703,32.5078) -- (41.4609,32.6641) .. controls (41.1797,32.9609) and
                    (40.7891,33.1055) .. (40.5195,33.1094) .. controls (40.2891,33.1133) and
                    (40.1719,33.0937) .. (40.0234,33.0234) .. controls (39.6211,32.8438) and
                    (39.4375,32.5195) .. (39.4375,32.0430) -- (39.4375,28.2188) .. controls
                    (39.4219,28.1641) and (39.3672,28.1328) .. (39.2812,28.1328) --
                    (38.0039,28.1328) .. controls (37.8711,28.1328) and (37.8047,28.1992) ..
                    (37.8047,28.3281) -- (37.8047,32.2266) .. controls (37.8047,32.7500) and
                    (37.8984,33.2031) .. (38.0859,33.5742) .. controls (38.2656,33.9297) and
                    (38.5156,34.1992) .. (38.8555,34.3867) .. controls (39.2031,34.5820) and
                    (39.6992,34.6797) .. (39.9258,34.6797) .. controls (41.0000,34.6797) and
                    (41.3164,34.1992) .. (41.7422,33.6836) .. controls (41.7422,33.6836) and
                    (41.9727,34.3516) .. (41.9883,34.3750) .. controls (42.0352,34.4492) and
                    (42.0898,34.4844) .. (42.1523,34.4844) -- (43.0039,34.4844) .. controls
                    (43.0547,34.4844) and (43.0977,34.4648) .. (43.1367,34.4258) .. controls
                    (43.1719,34.3867) and (43.1953,34.3477) .. (43.1953,34.3047) .. controls
                    (43.1953,34.3047) and (43.1992,28.2891) .. (43.1992,28.2617) .. controls
                    (43.1992,28.1758) and (43.1445,28.1328) .. (43.0351,28.1328);
                \path[fill=TUMWhite,nonzero rule] (42.5391,15.4102) .. controls (42.1484,15.1992)
                    and (41.6836,15.0938) .. (41.1445,15.0938) .. controls (40.6562,15.1094) and
                    (40.2461,15.2031) .. (39.9219,15.3672) -- (40.0273,14.3438) --
                    (42.8242,14.3438) .. controls (42.9336,14.3438) and (43.0000,14.2852) ..
                    (43.0195,14.1719) -- (43.2422,12.3047) .. controls (43.2422,12.1836) and
                    (43.1836,12.1211) .. (43.0664,12.1211) -- (38.1055,12.1211) .. controls
                    (37.9961,12.1211) and (37.9336,12.1836) .. (37.9180,12.3047) --
                    (37.5781,17.0586) .. controls (37.5625,17.1836) and (37.6211,17.2461) ..
                    (37.7578,17.2461) -- (40.2148,17.2461) .. controls (40.5234,17.2500) and
                    (40.7656,17.3047) .. (40.9336,17.4062) .. controls (41.1055,17.5117) and
                    (41.1914,17.6523) .. (41.1914,17.8359) .. controls (41.1914,18.0234) and
                    (41.1055,18.1719) .. (40.9375,18.2812) .. controls (40.7695,18.3867) and
                    (40.5312,18.4453) .. (40.2266,18.4453) .. controls (39.4414,18.4453) and
                    (38.7539,18.2656) .. (38.1523,17.9062) .. controls (38.1289,17.8906) and
                    (38.0898,17.8945) .. (38.0312,17.9140) .. controls (37.9766,17.9375) and
                    (37.9414,17.9648) .. (37.9297,18.0039) -- (37.3359,19.6992) .. controls
                    (37.3281,19.7148) and (37.3242,19.7383) .. (37.3242,19.7617) .. controls
                    (37.3242,19.8125) and (37.3555,19.8555) .. (37.4180,19.8906) .. controls
                    (38.3555,20.4570) and (39.3437,20.7383) .. (40.3828,20.7383) .. controls
                    (41.0625,20.7383) and (41.6602,20.6211) .. (42.1641,20.3828) .. controls
                    (42.6719,20.1445) and (43.0625,19.8086) .. (43.3320,19.3711) .. controls
                    (43.6055,18.9375) and (43.7422,18.4219) .. (43.7422,17.8359) .. controls
                    (43.7422,17.2461) and (43.6367,16.7500) .. (43.4297,16.3398) .. controls
                    (43.2227,15.9297) and (42.9258,15.6211) .. (42.5391,15.4101);
                \path[fill=TUMWhite,nonzero rule] (74.5508,20.5078) .. controls (74.5898,20.4688)
                    and (74.6055,20.4297) .. (74.6055,20.3789) -- (74.6055,15.9297) .. controls
                    (74.6055,15.4609) and (74.5156,15.0547) .. (74.3281,14.7188) .. controls
                    (74.1445,14.3828) and (73.8789,14.1250) .. (73.5352,13.9492) .. controls
                    (73.1875,13.7695) and (72.7773,13.6836) .. (72.3008,13.6836) .. controls
                    (71.5859,13.6836) and (71.0508,13.9102) .. (70.7031,14.3672) --
                    (70.7031,11.8984) .. controls (70.6992,11.8477) and (70.6797,11.8047) ..
                    (70.6445,11.7695) .. controls (70.6094,11.7344) and (70.5664,11.7148) ..
                    (70.5195,11.7148) -- (68.3672,11.7148) .. controls (68.3203,11.7109) and
                    (68.2774,11.7305) .. (68.2422,11.7695) .. controls (68.2070,11.8086) and
                    (68.1875,11.8516) .. (68.1875,11.8984) -- (68.1875,20.3789) .. controls
                    (68.1875,20.4336) and (68.2070,20.4766) .. (68.2422,20.5117) .. controls
                    (68.2774,20.5469) and (68.3203,20.5664) .. (68.3672,20.5664) --
                    (70.5195,20.5664) .. controls (70.5625,20.5664) and (70.6055,20.5469) ..
                    (70.6445,20.5039) .. controls (70.6836,20.4648) and (70.7031,20.4258) ..
                    (70.7031,20.3789) -- (70.7031,16.4648) -- (70.7500,16.2930) .. controls
                    (70.7852,16.1641) and (70.8750,16.0469) .. (71.0156,15.9414) .. controls
                    (71.1562,15.8320) and (71.3203,15.7813) .. (71.5078,15.7813) .. controls
                    (71.9062,15.7813) and (72.1055,16.0352) .. (72.1055,16.5469) --
                    (72.1055,20.3789) .. controls (72.0937,20.4297) and (72.1094,20.4687) ..
                    (72.1484,20.5078) .. controls (72.1836,20.5469) and (72.2305,20.5664) ..
                    (72.2773,20.5664) -- (74.4219,20.5664) .. controls (74.4687,20.5664) and
                    (74.5117,20.5469) .. (74.5508,20.5078);
                \path[fill=TUMWhite,nonzero rule] (71.6289,39.6914) -- (70.3516,39.6914) ..
                    controls (70.2461,39.6914) and (70.1797,39.7383) .. (70.1641,39.8242) --
                    (70.1641,48.1875) .. controls (70.1641,48.2734) and (70.2188,48.3164) ..
                    (70.3281,48.3164) -- (71.6055,48.3164) .. controls (71.7109,48.3164) and
                    (71.7734,48.2734) .. (71.7930,48.1836) -- (71.7930,39.8203) .. controls
                    (71.7930,39.7344) and (71.7383,39.6914) .. (71.6289,39.6914);
                \path[fill=TUMWhite,nonzero rule] (62.4453,42.4102) .. controls (62.4063,42.3906)
                    and (62.0859,42.2305) .. (62.0430,42.2109) .. controls (61.4688,41.9258) and
                    (60.8906,41.7812) .. (60.3047,41.7812) .. controls (59.5977,41.7812) and
                    (58.9688,41.9258) .. (58.4414,42.2070) .. controls (57.9141,42.4844) and
                    (57.5195,42.8672) .. (57.2305,43.3750) .. controls (56.9414,43.8750) and
                    (56.7969,44.4492) .. (56.7969,45.1406) .. controls (56.7969,45.8320) and
                    (56.9414,46.4180) .. (57.2305,46.9219) .. controls (57.5195,47.4258) and
                    (57.9141,47.8086) .. (58.4414,48.0898) .. controls (58.9688,48.3711) and
                    (59.5977,48.5117) .. (60.3047,48.5117) .. controls (60.9297,48.5117) and
                    (61.5508,48.3477) .. (62.1641,48.0234) .. controls (62.1641,48.0234) and
                    (62.2227,47.9961) .. (62.2422,47.9766) .. controls (62.2969,47.9336) and
                    (62.3047,47.8711) .. (62.2774,47.7812) -- (61.8984,47.0156) .. controls
                    (61.8399,46.9102) and (61.7578,46.8711) .. (61.6484,46.9062) .. controls
                    (61.6445,46.9062) and (61.4492,46.9922) .. (61.3438,47.0234) .. controls
                    (60.6797,47.2383) and (59.8281,47.1523) .. (59.3399,46.9023) .. controls
                    (59.0391,46.7461) and (58.7930,46.4687) .. (58.6133,46.1758) .. controls
                    (58.4297,45.8828) and (58.3399,45.5352) .. (58.3399,45.1445) .. controls
                    (58.3399,44.7578) and (58.4297,44.4141) .. (58.6133,44.1211) .. controls
                    (58.7930,43.8281) and (59.0391,43.5469) .. (59.3399,43.3945) .. controls
                    (59.8281,43.1445) and (60.7344,43.0547) .. (61.3984,43.2695) .. controls
                    (61.4492,43.2891) and (61.5234,43.3164) .. (61.5859,43.3437) .. controls
                    (61.6484,43.3672) and (61.9453,43.5195) .. (61.9453,43.5195) .. controls
                    (62.0547,43.5547) and (62.1406,43.5156) .. (62.1953,43.4102) --
                    (62.5547,42.6562) .. controls (62.5859,42.5664) and (62.5742,42.5039) ..
                    (62.5234,42.4570) .. controls (62.5039,42.4414) and (62.4453,42.4102) ..
                    (62.4453,42.4102);
                \path[fill=TUMWhite,nonzero rule] (64.7227,44.6562) .. controls (64.7227,43.8477)
                    and (65.3906,43.3359) .. (66.2305,43.3359) .. controls (67.0664,43.3359) and
                    (67.5859,43.8477) .. (67.5859,44.6562) -- cycle(66.2500,41.7930) .. controls
                    (65.7070,41.7930) and (64.3047,41.9531) .. (63.4297,43.3867) .. controls
                    (63.1211,43.8906) and (62.9688,44.5703) .. (62.9688,45.1484) .. controls
                    (62.9688,45.7305) and (63.1133,46.4414) .. (63.4414,46.9297) .. controls
                    (64.0898,47.9023) and (65.0781,48.5078) .. (66.3438,48.5117) .. controls
                    (67.2266,48.5156) and (68.3555,47.9648) .. (68.3516,47.9609) --
                    (68.3555,47.9648) .. controls (68.4687,47.9141) and (68.5078,47.8281) ..
                    (68.4687,47.7070) -- (68.1406,46.8555) .. controls (68.0781,46.7383) and
                    (67.9883,46.7070) .. (67.8711,46.7617) .. controls (67.8711,46.7617) and
                    (67.8672,46.7617) .. (67.8633,46.7617) .. controls (67.2109,46.9609) and
                    (67.0469,47.1016) .. (66.3242,47.1016) .. controls (65.4336,47.1016) and
                    (64.7227,46.4727) .. (64.7227,45.7578) -- (68.7031,45.7578) .. controls
                    (68.8203,45.7422) and (68.8750,45.7148) .. (68.9102,45.5508) .. controls
                    (68.9727,45.2500) and (68.9688,44.8984) .. (68.9688,44.7070) .. controls
                    (68.9688,44.0273) and (68.8086,43.5273) .. (68.5430,43.0820) .. controls
                    (67.9219,42.0547) and (66.7930,41.7930) .. (66.2500,41.7930);
            \end{scope}%
        \end{tikzpicture}%
    }%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMlogo.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Commands ---
%================
\newcommand{\AMlogoAM}[4][!]{%
    \resizebox{#1}{#2}{%
        \AMtikzExternalizeSkipNext%
        \begin{tikzpicture}[y=0.80pt, x=0.80pt, yscale=-1, xscale=1, inner sep=0pt, outer sep=0pt]%
            \begin{scope}[shift={(0,-1016.9291)}]%
                \begin{scope}[cm={{0.85655,0.0,0.0,0.85655,(-0.55087,151.52721)}}]%
                \path[fill=#4,even odd rule] (35.3293,1049.0472) .. controls
                    (33.5002,1048.2521) and (32.2066,1047.3762) .. (31.0238,1046.3603) .. controls
                    (29.2791,1046.8050) and (27.5074,1047.3170) .. (25.7960,1047.8765) .. controls
                    (24.8142,1048.1975) and (23.8529,1048.5338) .. (22.9284,1048.8818) .. controls
                    (23.4872,1049.4012) and (24.0733,1049.8576) .. (24.6898,1050.2399) --
                    (35.3293,1050.2549) -- (35.3293,1049.0470) -- cycle(30.3531,1045.7608) ..
                    controls (29.7790,1045.2283) and (29.2128,1044.6585) .. (28.6020,1044.0441) ..
                    controls (27.5113,1042.9467) and (26.2747,1041.7027) .. (24.5881,1040.2708) ..
                    controls (22.6382,1040.4455) and (20.3088,1040.8781) .. (17.8512,1041.4578) ..
                    controls (18.7335,1043.2433) and (19.6827,1044.9143) .. (20.7362,1046.3653) ..
                    controls (21.2460,1047.0676) and (21.7800,1047.7178) .. (22.3420,1048.3038) ..
                    controls (23.3763,1047.9077) and (24.4581,1047.5261) .. (25.5643,1047.1643) ..
                    controls (27.1304,1046.6523) and (28.7486,1046.1793) .. (30.3531,1045.7608) --
                    cycle(23.8068,1039.5990) .. controls (22.0075,1039.2323) and
                    (20.2217,1038.4864) .. (18.4706,1036.7923) .. controls (18.3232,1036.6499) and
                    (18.1783,1036.5121) .. (18.0351,1036.3792) .. controls (17.6733,1037.3674) and
                    (17.4001,1038.3291) .. (17.2372,1039.2450) .. controls (17.1947,1039.4835) and
                    (17.1596,1039.7188) .. (17.1326,1039.9505) .. controls (17.2591,1040.2251) and
                    (17.3867,1040.4976) .. (17.5157,1040.7679) .. controls (19.7713,1040.2315) and
                    (21.9299,1039.8144) .. (23.8068,1039.5990) -- cycle(15.4106,1034.3103) ..
                    controls (14.1672,1033.5339) and (13.0646,1033.1771) .. (11.9860,1033.1155) ..
                    controls (12.6890,1033.4925) and (13.3409,1034.0357) .. (13.9472,1034.7025) ..
                    controls (14.2895,1035.0788) and (14.6172,1035.4951) .. (14.9315,1035.9430) ..
                    controls (15.0725,1035.4020) and (15.2323,1034.8570) .. (15.4106,1034.3103) --
                    cycle(35.3293,1045.3923) .. controls (34.2309,1045.6046) and
                    (33.0878,1045.8544) .. (31.9256,1046.1360) .. controls (32.8847,1046.9103) and
                    (33.9467,1047.5977) .. (35.3293,1048.2277) -- (35.3293,1045.3923) --
                    cycle(35.3293,1042.2905) .. controls (32.0755,1039.8254) and
                    (28.7192,1040.1943) .. (25.3899,1039.8378) -- (25.3969,1039.9788) .. controls
                    (26.9386,1041.3104) and (28.1001,1042.4786) .. (29.1319,1043.5168) .. controls
                    (29.8636,1044.2528) and (30.5294,1044.9225) .. (31.2225,1045.5390) .. controls
                    (32.6256,1045.1889) and (34.0097,1044.8828) .. (35.3292,1044.6312) --
                    (35.3292,1042.2905) -- cycle(10.0736,1033.2909) .. controls (8.5720,1033.6278)
                    and (6.9846,1034.3861) .. (5.0064,1035.2443) .. controls (7.0806,1036.3381)
                    and (8.4770,1039.6118) .. (9.6562,1042.9573) .. controls (11.1575,1042.4868)
                    and (12.7102,1042.0265) .. (14.2583,1041.6035) .. controls (14.2211,1040.8742)
                    and (14.2413,1040.1220) .. (14.3136,1039.3524) .. controls (14.3880,1038.5615)
                    and (14.5173,1037.7533) .. (14.6949,1036.9337) .. controls (14.2852,1036.2945)
                    and (13.8523,1035.7109) .. (13.3938,1035.2064) .. controls (12.4112,1034.1257)
                    and (11.3110,1033.4103) .. (10.0737,1033.2909) -- cycle(9.9025,1043.6646) ..
                    controls (10.0581,1044.1174) and (10.2104,1044.5688) .. (10.3607,1045.0139) ..
                    controls (11.0668,1047.1066) and (11.7216,1049.0476) .. (12.4168,1050.2225) --
                    (17.9639,1050.2325) .. controls (19.0485,1049.6738) and (20.2745,1049.1260) ..
                    (21.5908,1048.6001) .. controls (21.0813,1048.0488) and (20.5951,1047.4482) ..
                    (20.1296,1046.8069) .. controls (19.0219,1045.2811) and (18.0277,1043.5197) ..
                    (17.1060,1041.6393) -- (17.0837,1041.6493) .. controls (17.0993,1041.8644) and
                    (17.1241,1042.0749) .. (17.1585,1042.2800) .. controls (17.3222,1043.2608) and
                    (17.1655,1044.6558) .. (16.8336,1045.4222) .. controls (16.7096,1045.7082) and
                    (16.5519,1045.9300) .. (16.3631,1046.0512) -- (16.3634,1046.0519) .. controls
                    (16.0941,1046.2248) and (15.8075,1046.2177) .. (15.5187,1045.9719) .. controls
                    (15.2140,1045.7118) and (14.8951,1045.0807) .. (14.5982,1043.9196) .. controls
                    (14.4702,1043.4193) and (14.3781,1042.9020) .. (14.3193,1042.3701) .. controls
                    (12.8329,1042.7786) and (11.3436,1043.2201) .. (9.9026,1043.6716) --
                    cycle(23.3749,1050.2381) .. controls (22.9539,1049.9164) and
                    (22.5482,1049.5617) .. (22.1560,1049.1787) .. controls (21.2833,1049.5213) and
                    (20.4492,1049.8739) .. (19.6689,1050.2328) -- (23.3749,1050.2328) --
                    cycle(4.0171,1035.6659) .. controls (3.4271,1035.9125) and (2.8024,1036.1627)
                    .. (2.1370,1036.4097) -- (2.1370,1045.5532) .. controls (4.0366,1044.8275) and
                    (6.4003,1043.9952) .. (8.9430,1043.1831) .. controls (7.6840,1039.6093) and
                    (6.1909,1036.1517) .. (4.0171,1035.6659) -- cycle(1.3993,1036.6758) --
                    (1.3933,1036.6758) -- (1.3933,1045.8391) -- (1.3993,1045.8391) --
                    (1.3993,1036.6761) -- cycle(11.5636,1050.2211) .. controls (10.9049,1048.9661)
                    and (10.2994,1047.1710) .. (9.6513,1045.2513) .. controls (9.5000,1044.8027)
                    and (9.3462,1044.3470) .. (9.1889,1043.8903) .. controls (6.5318,1044.7382)
                    and (4.0713,1045.6088) .. (2.1370,1046.3550) -- (2.1370,1050.2077) --
                    (11.5636,1050.2217) -- cycle(1.3993,1046.6438) -- (1.3933,1046.6438) --
                    (1.3933,1050.9542) -- (1.7685,1050.9542) -- (1.3997,1050.9539) --
                    (1.3997,1050.5794) -- (1.3997,1046.6413) -- cycle;
                \path[fill=#3,nonzero rule] (36.8235,1034.2149) -- (36.8235,1051.3316) --
                    (36.8235,1051.7065) -- (36.4486,1051.7065) -- (1.0180,1051.7065) --
                    (0.6431,1051.7065) -- (0.6431,1051.3316) -- (0.6431,1032.1435) --
                    (0.6431,1015.9010) -- (0.6431,1015.5261) -- (1.0180,1015.5261) --
                    (25.9438,1015.5261) .. controls (26.0405,1015.7625) and (26.1503,1016.0020) ..
                    (26.2726,1016.2433) -- (26.2726,1016.2763) -- (1.3929,1016.2763) --
                    (1.3929,1032.1439) -- (1.3929,1050.9571) -- (36.0737,1050.9571) --
                    (36.0737,1034.5080) -- (36.0850,1034.5080) -- (36.0967,1034.5080) --
                    (36.1047,1034.5070) -- (36.1087,1034.5060) -- (36.1200,1034.5060) --
                    (36.1313,1034.5060) -- (36.1426,1034.5060) -- (36.1536,1034.5060) --
                    (36.1649,1034.5060) -- (36.1759,1034.5060) -- (36.1862,1034.5060) --
                    (36.1872,1034.5057) -- (36.1982,1034.5057) -- (36.2092,1034.5057) --
                    (36.2202,1034.5057) -- (36.2252,1034.5057) -- (36.2302,1034.5047) --
                    (36.2412,1034.5047) -- (36.2518,1034.5047) -- (36.2734,1034.4947) --
                    (36.2840,1034.4947) -- (36.2946,1034.4947) -- (36.3026,1034.4947) --
                    (36.3056,1034.4940) -- (36.3162,1034.4940) -- (36.3265,1034.4940) --
                    (36.3371,1034.4940) -- (36.3401,1034.4940) -- (36.3471,1034.4940) --
                    (36.3574,1034.4940) -- (36.3677,1034.4940) -- (36.3777,1034.4940) --
                    (36.3781,1034.4940) -- (36.3884,1034.4940) -- (36.3987,1034.4940) --
                    (36.4087,1034.4940) -- (36.4147,1034.4940) -- (36.4197,1034.4940) --
                    (36.4297,1034.4940) -- (36.4397,1034.4940) -- (36.4497,1034.4940) --
                    (36.4507,1034.4933) -- (36.4587,1034.4933) -- (36.4687,1034.4933) --
                    (36.4787,1034.4933) -- (36.4857,1034.4933) -- (36.4877,1034.4933) --
                    (36.4977,1034.4933) -- (36.5077,1034.4933) -- (36.5177,1034.4933) --
                    (36.5217,1034.4933) -- (36.5277,1034.4933) -- (36.5377,1034.4933) --
                    (36.5477,1034.4933) -- (36.5567,1034.4933) -- (36.5567,1034.4929) --
                    (36.5667,1034.4929) -- (36.5757,1034.4929) -- (36.5847,1034.4929) --
                    (36.5897,1034.4929) -- (36.5937,1034.4929) -- (36.6027,1034.4929) --
                    (36.6117,1034.4929) -- (36.6207,1034.4929) -- (36.6227,1034.4919) --
                    (36.6297,1034.4919) -- (36.6387,1034.4919) -- (36.6477,1034.4919) --
                    (36.6547,1034.4919) -- (36.6567,1034.4909) -- (36.6657,1034.4809) --
                    (36.6747,1034.4809) -- (36.6837,1034.4709) -- (36.6867,1034.4709) --
                    (36.6917,1034.4709) -- (36.6997,1034.4609) -- (36.7087,1034.4609) --
                    (36.7167,1034.4509) -- (36.7177,1034.4509) -- (36.7207,1034.4509) --
                    (36.7237,1034.4509) -- (36.7277,1034.4509) -- (36.7307,1034.4509) --
                    (36.7347,1034.4509) -- (36.7387,1034.4509) -- (36.7417,1034.4509) --
                    (36.7437,1034.4509) -- (36.7444,1034.4505) -- (36.7484,1034.4505) --
                    (36.7514,1034.4505) -- (36.7544,1034.4505) -- (36.7564,1034.4495) --
                    (36.7574,1034.4488) -- (36.7614,1034.4488) -- (36.7644,1034.4488) --
                    (36.7674,1034.4488) -- (36.7684,1034.4478) -- (36.7704,1034.4468) --
                    (36.7734,1034.4468) -- (36.7764,1034.4468) -- (36.7794,1034.4468) --
                    (36.7804,1034.4461) -- (36.7824,1034.4451) -- (36.7854,1034.4451) --
                    (36.7884,1034.4451) -- (36.7914,1034.4451) -- (36.7921,1034.4448) --
                    (36.7951,1034.4448) -- (36.7981,1034.4448) -- (36.8011,1034.4448) --
                    (36.8041,1034.4448) -- (36.8045,1034.4448) -- (36.8075,1034.4448) --
                    (36.8115,1034.4448) -- (36.8145,1034.4448) -- (36.8175,1034.4448) --
                    (36.8178,1034.4444) -- (36.8208,1034.4444) -- cycle;
                \path[fill=#3,even odd rule] (17.5603,1035.5274) .. controls
                    (17.2935,1035.3814) and (17.0221,1035.2276) .. (16.7482,1035.0675) --
                    (16.7475,1035.0685) .. controls (16.5257,1034.9388) and (16.3035,1034.8048) ..
                    (16.0820,1034.6677) .. controls (15.5594,1036.2955) and (15.2008,1037.8978) ..
                    (15.0577,1039.4211) -- (15.0577,1039.4221) .. controls (14.9138,1040.9496) and
                    (14.9858,1042.4038) .. (15.3245,1043.7279) .. controls (15.5746,1044.7062) and
                    (15.8010,1045.2108) .. (15.9938,1045.3865) .. controls (16.0413,1045.3365) and
                    (16.0937,1045.2427) .. (16.1476,1045.1190) .. controls (16.4314,1044.4631) and
                    (16.5636,1043.2549) .. (16.4201,1042.3963) .. controls (16.2514,1041.3897) and
                    (16.2907,1040.2845) .. (16.4988,1039.1156) .. controls (16.7018,1037.9750) and
                    (17.0682,1036.7674) .. (17.5603,1035.5269) -- cycle;
                \path[fill=#3,even odd rule] (36.6761,1032.8993) .. controls
                    (36.5616,1032.4033) and (36.0135,1031.8193) .. (34.9207,1031.3438) .. controls
                    (33.7819,1030.8485) and (32.9312,1030.8499) .. (32.4617,1031.1167) .. controls
                    (32.3175,1031.1987) and (32.2154,1031.3059) .. (32.1612,1031.4278) .. controls
                    (32.1081,1031.5472) and (32.0974,1031.6910) .. (32.1336,1031.8473) --
                    (32.1336,1031.8473) .. controls (32.2491,1032.3459) and (32.7997,1032.9337) ..
                    (33.8925,1033.4092) .. controls (35.0232,1033.9014) and (35.8736,1033.8978) ..
                    (36.3452,1033.6299) -- (36.3448,1033.6292) .. controls (36.4901,1033.5462) and
                    (36.5929,1033.4390) .. (36.6471,1033.3178) .. controls (36.7006,1033.1980) and
                    (36.7123,1033.0542) .. (36.6761,1032.8979) -- cycle;
                \path[fill=#3,even odd rule] (36.7633,1031.6592) .. controls
                    (35.5865,1028.7919) and (32.2427,1025.3779) .. (28.1250,1022.8360) .. controls
                    (27.8919,1023.0103) and (27.6598,1023.1924) .. (27.4284,1023.3816) --
                    (27.3515,1023.4726) .. controls (30.2638,1025.5146) and (32.4447,1028.2823) ..
                    (32.4670,1030.3066) .. controls (33.1285,1030.1022) and (34.0764,1030.1610) ..
                    (35.2198,1030.6581) .. controls (35.8941,1030.9515) and (36.4032,1031.2980) ..
                    (36.7632,1031.6595) -- cycle;
                \path[fill=#3,even odd rule] (30.6883,1034.9367) .. controls
                    (30.9292,1034.5909) and (23.1750,1027.8749) .. (22.5344,1029.7989) .. controls
                    (22.5082,1029.8769) and (22.4231,1030.9331) .. (22.3976,1031.3732) .. controls
                    (21.6645,1031.1114) and (20.9204,1030.7921) .. (20.1792,1030.4151) .. controls
                    (15.3036,1027.9373) and (12.3743,1023.9107) .. (13.6386,1021.4233) .. controls
                    (14.6523,1019.4284) and (18.8504,1019.6644) .. (22.7580,1020.9297) .. controls
                    (22.9688,1020.6551) and (23.1810,1020.3851) .. (23.3944,1020.1211) .. controls
                    (16.8353,1017.6078) and (9.9298,1017.5373) .. (8.2294,1020.8843) .. controls
                    (6.2816,1024.7168) and (10.7947,1030.9218) .. (18.3079,1034.7401) .. controls
                    (19.5449,1035.3686) and (20.7865,1035.8934) .. (22.0057,1036.3144) .. controls
                    (21.9667,1036.7562) and (21.8923,1037.4585) .. (21.9316,1037.6211) .. controls
                    (22.4543,1039.7662) and (30.1000,1035.7807) .. (30.6882,1034.9367) -- cycle;
                \path[fill=#3,even odd rule] (18.9776,1028.2041) .. controls
                    (19.2958,1028.4355) and (19.6236,1028.6594) .. (19.9595,1028.8734) --
                    (19.9602,1028.8724) .. controls (20.3000,1029.0886) and (20.6469,1029.2958) ..
                    (20.9991,1029.4914) .. controls (22.3969,1027.4370) and (23.9390,1025.5817) ..
                    (25.5497,1024.0457) .. controls (26.9993,1022.6635) and (28.5081,1021.5356) ..
                    (30.0235,1020.7490) .. controls (29.8524,1020.6080) and (29.6809,1020.4595) ..
                    (29.5108,1020.3040) -- (29.5101,1020.3047) .. controls (29.3216,1020.1325) and
                    (29.1331,1019.9504) .. (28.9464,1019.7587) .. controls (28.6849,1019.5132) and
                    (28.4280,1019.2520) .. (28.1775,1018.9756) -- (28.1768,1018.9763) --
                    (28.1760,1018.9756) -- (28.1720,1018.9756) -- (28.1561,1018.9556) .. controls
                    (27.6877,1018.4369) and (27.2742,1017.8997) .. (26.9195,1017.3601) .. controls
                    (25.5447,1018.6835) and (24.1763,1020.2599) .. (22.8628,1022.0362) .. controls
                    (21.4901,1023.8929) and (20.1766,1025.9700) .. (18.9776,1028.2079) -- cycle;
                \path[fill=#3,even odd rule] (28.7061,1018.4455) -- (28.7068,1018.4462) --
                    (28.7185,1018.4602) .. controls (29.9300,1019.8002) and (31.2998,1020.7623) ..
                    (32.5340,1021.2324) .. controls (33.6870,1021.6718) and (34.6975,1021.6757) ..
                    (35.2907,1021.1394) .. controls (35.3587,1021.0774) and (35.4214,1021.0087) ..
                    (35.4774,1020.9343) .. controls (35.5298,1020.8643) and (35.5770,1020.7876) ..
                    (35.6177,1020.7075) -- (35.7020,1020.4634) -- (35.7013,1020.4630) --
                    (35.7931,1020.1891) -- (35.7924,1020.1888) .. controls (36.2112,1018.9235) and
                    (36.5748,1017.6093) .. (36.9358,1016.3028) .. controls (37.3054,1014.9656) and
                    (37.6739,1013.6337) .. (38.0913,1012.3815) -- (38.0913,1012.3805) .. controls
                    (38.5115,1011.1205) and (38.6132,1010.5574) .. (38.2050,1010.4214) .. controls
                    (37.6643,1010.2410) and (36.4642,1010.3804) .. (34.4296,1010.6155) .. controls
                    (32.7529,1010.8097) and (30.5221,1011.0687) .. (27.5978,1011.2618) .. controls
                    (27.4235,1011.2858) and (27.2594,1011.3318) .. (27.1088,1011.3979) --
                    (27.1092,1011.3982) .. controls (26.9674,1011.4602) and (26.8399,1011.5403) ..
                    (26.7290,1011.6378) -- (26.7297,1011.6385) -- (26.7293,1011.6388) --
                    (26.7148,1011.6508) .. controls (26.1213,1012.1873) and (26.0235,1013.1932) ..
                    (26.3438,1014.3845) .. controls (26.6861,1015.6548) and (27.5007,1017.1097) ..
                    (28.7055,1018.4462) -- (28.7062,1018.4455) -- cycle;
                \end{scope}%
            \end{scope}%
        \end{tikzpicture}%
    }%
}%
%
%
%
\else\fi% End-check if module AMlogo is enabled
%###################################################################################################
%--- END MODULE AMlogo -----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMmath ---------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMmath% Start-check if module AMmath is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMmath.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%===============================
%--- Default Option handling ---
%===============================
% Select left aligned equations, if no other option is given
\ifdefined\optLeftEquations\else%
\ifdefined\optCenterEquations\else%
    \def\optLeftEquations{x}%
\fi%
\fi%
%
%
%================
%--- Packages ---
%================
\ifdefined\optLeftEquations%
    \PassOptionsToPackage{fleqn}{amsmath}%
\fi%
\RequirePackage{amsmath}%
\RequirePackage{mathtools}%
\RequirePackage{siunitx}%
\ifdefined\optComputerModernMath% in this case the mathdesign package has not been loaded
    \RequirePackage{amssymb}%
    \RequirePackage{amsfonts}%
\else\fi%
%
\ifdefined\optLimitMathFonts\else%
\newcommand\hmmax{0}%
\newcommand\bmmax{2}% limit number of additional math fonts, while still enabling correct spacing of bold Greek math letters with indices
\RequirePackage{bm}%
\fi%
%
%===================
%--- Definitions ---
%===================
% Math operators
% --------------
\DeclareMathOperator{\real}{Re}%
\DeclareMathOperator{\imag}{Im}%
\DeclareMathOperator{\asin}{asin}%
\DeclareMathOperator{\acos}{acos}%
\DeclareMathOperator{\atan}{atan}%
\DeclareMathOperator{\co}{co}%
\DeclareMathOperator{\atanII}{atan2}%
\DeclareMathOperator{\AtanII}{Atan2}%
\DeclareMathOperator{\sinabr}{s}%
\DeclareMathOperator{\cosabr}{c}%
\DeclareMathOperator{\sign}{sign}%
\let\sgn\relax%
\DeclareMathOperator{\sgn}{sgn}%
\DeclareMathOperator*{\argmin}{arg\,min}%
\DeclareMathOperator*{\argmax}{arg\,max}%
\let\div\undefined%
\DeclareMathOperator{\div}{div}%
\DeclareMathOperator{\grad}{grad}%
\DeclareMathOperator{\curl}{curl}%
\DeclareMathOperator{\rot}{rot}%
\DeclareMathOperator{\dif}{d \!}%
\DeclareMathOperator{\Dif}{D \!}%
\DeclareMathOperator{\order}{O}%
%
% Misc
% ----
\newcommand{\Abs}[1]{\mathinner{\left\vert#1\right\vert}}%
\newcommand{\abs}[1]{\mathinner{\vert#1\vert}}%
\newcommand{\Norm}[1]{\mathinner{\left\Vert#1\right\Vert}}%
\newcommand{\norm}[1]{\mathinner{\Vert#1\Vert}}%
\providecommand{\e}{\ensuremath{\mathalpha{\mathrm{e}}}}%
\newcommand{\const}{\mbox{const}}%
\newcommand{\konst}{\mbox{konst}}%
\newcommand{\laplacian}{\Delta}%
\newcommand{\vdiv}{\vnabla\cdot}%
\newcommand{\vgrad}{\vnabla}%
\newcommand{\vcurl}{\vnabla\times}%
\newcommand{\vrot}{\vcurl}%
%
% Numeral system
% --------------
\newcommand{\hex}[1]{\mathalpha{\mathtt{#1}_\mathrm{H}}}%
\newcommand{\bin}[1]{\mathalpha{\mathtt{#1}_\mathrm{B}}}%
\newcommand{\dec}[1]{\mathalpha{\mathtt{#1}_\mathrm{D}}}%
\newcommand{\bnot}[1]{\mathalpha{\overline{\mbox{$#1$}}}}%
%
% Sets of numbers
% ---------------
\newcommand{\MR}{\ensuremath{\mathalpha{\mathbb{R}}}}%
\newcommand{\MN}{\ensuremath{\mathalpha{\mathbb{N}}}}%
\newcommand{\MZ}{\ensuremath{\mathalpha{\mathbb{Z}}}}%
\newcommand{\MC}{\ensuremath{\mathalpha{\mathbb{C}}}}%
\newcommand{\MQ}{\ensuremath{\mathalpha{\mathbb{Q}}}}%
\newcommand{\Mone}{\ensuremath{\mathalpha{\mathbb{1}}}}%
%
% Relations
% ---------
\newcommand{\eqhat}{\mathrel{\widehat{=}}}%
\newcommand{\eqexcl}{\stackrel{!}{=}}%
\newcommand{\eqdef}{\stackrel{\mathrm{def.}}{=}}%
\newcommand{\defined}{\coloneqq}%
\newcommand{\rdefined}{\eqqcolon}%
%
% Vectors and matrices
% --------------------
% vectors (main definition)
\let\origvec\vec%
\let\vec\relax%
\newcommand{\vec}[1]{\ensuremath{\boldsymbol{#1}}}%
%
% vectors (latin letters lower case)
\def\vecdeflatinlower#1{\expandafter\def\csname v#1\endcsname{\vec{#1}}}%
\def\vecdeflatinlowerall#1{\ifx#1\vecdeflatinlowerall\else\vecdeflatinlower{#1}\expandafter\vecdeflatinlowerall\fi}%
\vecdeflatinlowerall abcdefghijklmnopqrstuvwxyz\vecdeflatinlowerall%
%
% vectors (latin letters upper case)
\def\vecdeflatinupper#1{\expandafter\def\csname v#1\endcsname{\vec{#1}}}%
\def\vecdeflatinupperall#1{\ifx#1\vecdeflatinupperall\else\vecdeflatinupper{#1}\expandafter\vecdeflatinupperall\fi}%
\vecdeflatinupperall ABCDEFGHIJKLMNOPQRSTUVWXYZ\vecdeflatinupperall%
%
% vectors (greek letters lower case)
\def\vecdefgreeklower#1{\expandafter\def\csname v#1\endcsname{\vec{\csname #1\endcsname}}}%
\def\vecdefgreeklowerall#1{\ifx#1\vecdefgreeklowerall\else\vecdefgreeklower{#1}\expandafter\vecdefgreeklowerall\fi}%
\vecdefgreeklowerall {alpha}{beta}{gamma}{delta}{epsilon}{varepsilon}{zeta}{eta}{theta}{vartheta}{iota}{kappa}{lambda}{mu}{nu}{pi}{varpi}%
{xi}{rho}{varrho}{sigma}{varsigma}{tau}{upsilon}{phi}{varphi}{chi}{psi}{omega}\vecdefgreeklowerall%
%
% vectors (greek letters upper case)
\def\vecdefgreekupper#1{\expandafter\def\csname v#1\endcsname{\vec{\csname #1\endcsname}}}%
\def\vecdefgreekupperall#1{\ifx#1\vecdefgreekupperall\else\vecdefgreekupper{#1}\expandafter\vecdefgreekupperall\fi}%
\vecdefgreekupperall {Gamma}{Delta}{Theta}{Lambda}{Xi}{Pi}{Sigma}{Upsilon}{Phi}{Psi}{Omega}\vecdefgreekupperall%
%
% vectors (misc)
\newcommand{\vnull}{\vec{0}}%
\newcommand{\vzero}{\vec{0}}%
\newcommand{\vone}{\vec{1}}%
\newcommand{\vnabla}{\vec{\nabla}}%
%
% matrices
\newcommand{\mat}[1]{\begin{matrix}#1\end{matrix}}%
\newcommand{\pmat}[1]{\begin{pmatrix}#1\end{pmatrix}}%
\newcommand{\bmat}[1]{\begin{bmatrix}#1\end{bmatrix}}%
\newcommand{\Bmat}[1]{\begin{Bmatrix}#1\end{Bmatrix}}%
\newcommand{\vmat}[1]{\begin{vmatrix}#1\end{vmatrix}}%
\newcommand{\Vmat}[1]{\begin{Vmatrix}#1\end{Vmatrix}}%
%
% Derivatives
% -----------
\newcommand{\dd}[1]{\dif #1}%
\newcommand{\vdot}[1]{\vec{\dot{#1}}}%
\newcommand{\vddot}[1]{\vec{\ddot{#1}}}%
\newcommand{\diff}[3][]{\frac{\mathrm{d}^{#1}\,\!#2}{\mathrm{d}\,\!{#3}^{#1}}}%
\newcommand{\tdiff}[3][]{\tfrac{\mathrm{d}^{#1}\,\!#2}{\mathrm{d}\,\!{#3}^{#1}}}%
\newcommand{\pdiff}[3][]{\frac{\partial^{#1}\,\!#2}{\partial\,\!#3^{#1}}}%
\newcommand{\tpdiff}[3][]{\tfrac{\partial^{#1}\,\!#2}{\partial\,\!#3^{#1}}}%
\newcommand{\Ddiff}[3][]{\frac{D^{#1}\,\!#2}{D\,\!#3^{#1}}}%
\newcommand{\tDdiff}[3][]{\frac{D^{#1}\,\!#2}{D\,\!#3^{#1}}}%
%
% Accents
% -------
% Declare new symbol font for non-bold accents
\DeclareSymbolFont{nbaccents}{OT1}{cmr}{m}{n}%
% Set accents in math mode to use non-bold font
\DeclareMathAccent{\acute}{\mathord}{nbaccents}{"13}%
\DeclareMathAccent{\bar}{\mathord}{nbaccents}{"16}%
\DeclareMathAccent{\breve}{\mathord}{nbaccents}{"15}%
\DeclareMathAccent{\check}{\mathord}{nbaccents}{"14}%
\DeclareMathAccent{\dot}{\mathord}{nbaccents}{"5F}%
\DeclareMathAccent{\ddot}{\mathord}{nbaccents}{"7F}%
\DeclareMathAccent{\grave}{\mathord}{nbaccents}{"12}%
\DeclareMathAccent{\hat}{\mathord}{nbaccents}{"5E}%
\DeclareMathAccent{\tilde}{\mathord}{nbaccents}{"7E}%
%
%
% Indices
% -------
\def\rs#1{\@ifnextchar[%]
  {\@rs{#1}}{\@@rs{#1}}}%
\def\@rs#1[#2]#3{\mathinner{%
    \setbox\@ne\hbox{$\displaystyle{\vphantom{#3}}#1{#3}\m@th$}%
    \setbox\tw@\hbox{$\displaystyle{#3}#2\m@th$}%
    \hskip\wd\@ne\hskip-\wd\tw@\mathord{\hskip\wd\tw@\hskip-\wd\@ne%
      {\vphantom{#3}}#1{#3}#2}}}%
\def\@@rs#1#2{\mathinner{%
    \setbox\@ne\hbox{$\displaystyle{\vphantom{#2}}#1{#2}\m@th$}%
    \hskip\wd\@ne\mathord{\hskip-\wd\@ne%
      {\vphantom{#2}}#1{#2}}}}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMmath.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%===================
%--- Definitions ---
%===================
% Math operators
% --------------
\DeclareMathOperator{\prox}{prox}%
\DeclareMathOperator{\vprox}{\mathbf{prox}}%
\DeclareMathOperator{\proj}{proj}%
%
% Misc
% ----
\newcommand{\foeppl}[1]{\langle #1 \rangle}%
%
%
%
\else\fi% End-check if module AMmath is enabled
%###################################################################################################
%--- END MODULE AMmath -----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMlayout -------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMlayout% Start-check if module AMlayout is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMlayout.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%===============================
%--- Default Option Handling ---
%===============================
% Select A4-size, if no option has been passed
\ifdefined\optAfive\else%
\ifdefined\optAfour\else%
\ifdefined\optAthree\else%
\ifdefined\optAtwo\else%
\ifdefined\optAone\else%
\ifdefined\optAzero\else%
\ifdefined\optBeamerClassicFormat\else%
\ifdefined\optBeamerWideFormat\else%
    \def\optAfour{x}%
\fi%
\fi%
\fi%
\fi%
\fi%
\fi%
\fi%
\fi%
%
%================
%--- Packages ---
%================
\RequirePackage{geometry}%
%
%
%=============
%--- Style ---
%=============
%
%
%================
%--- Commands ---
%================
% Set indent
% ----------
\newcommand{\AMlayoutSetIndent}{%
    \ifdefined\optEnglish%
        % English
    	\parindent1.5em%
    	\parskip0em%
    \else%
        % German
    	\parindent0em%
    	\parskip0.5em%
    \fi%
}%
%
\newcommand{\AMlayoutNoIndent}{%
	\parindent0em%
	\parskip0em%
}%
%
% Set document format
% -------------------
% Header size
\newlength{\AMlayoutHeaderSize}% Factor to specify the scaling of header elements for the current format
%
\newcommand{\AMlayoutSetAfive}{%
    \def\optAfive{x}%
    \undef\optAfour%
    \undef\optAthree%
    \undef\optAtwo%
    \undef\optAone%
    \undef\optAzero%
    \undef\optBeamerClassicFormat%
    \undef\optBeamerWideFormat%
    \geometry{a5paper, margin=1.41cm}% (2/sqrt(2))
    \setlength{\AMlayoutHeaderSize}{0.71cm}% (1/sqrt(2))
}%
\newcommand{\AMlayoutSetAfour}{%
    \undef\optAfive%
    \def\optAfour{x}%
    \undef\optAthree%
    \undef\optAtwo%
    \undef\optAone%
    \undef\optAzero%
    \undef\optBeamerClassicFormat%
    \undef\optBeamerWideFormat%
    \geometry{a4paper, margin=2cm}%
    \setlength{\AMlayoutHeaderSize}{1cm}% (default according to TUM Corporate Design)
}%
\newcommand{\AMlayoutSetAthree}{%
    \undef\optAfive%
    \undef\optAfour%
    \def\optAthree{x}%
    \undef\optAtwo%
    \undef\optAone%
    \undef\optAzero%
    \undef\optBeamerClassicFormat%
    \undef\optBeamerWideFormat%
    \geometry{a3paper, margin=2.82cm}% (2*sqrt(2))
    \setlength{\AMlayoutHeaderSize}{1.41cm}% (sqrt(2))
}%
\newcommand{\AMlayoutSetAtwo}{%
    \undef\optAfive%
    \undef\optAfour%
    \undef\optAthree%
    \def\optAtwo{x}%
    \undef\optAone%
    \undef\optAzero%
    \undef\optBeamerClassicFormat%
    \undef\optBeamerWideFormat%
    \geometry{a2paper, margin=4cm}% (2*sqrt(2)*sqrt(2))
    \setlength{\AMlayoutHeaderSize}{2cm}% (sqrt(2)*sqrt(2))
}%
\newcommand{\AMlayoutSetAone}{%
    \undef\optAfive%
    \undef\optAfour%
    \undef\optAthree%
    \undef\optAtwo%
    \def\optAone{x}%
    \undef\optAzero%
    \undef\optBeamerClassicFormat%
    \undef\optBeamerWideFormat%
    \geometry{a1paper, margin=5.66cm}% (2*sqrt(2)*sqrt(2)*sqrt(2))
    \setlength{\AMlayoutHeaderSize}{2.83cm}% (sqrt(2)*sqrt(2)*sqrt(2))
}%
\newcommand{\AMlayoutSetAzero}{%
    \undef\optAfive%
    \undef\optAfour%
    \undef\optAthree%
    \undef\optAtwo%
    \undef\optAone%
    \def\optAzero{x}%
    \undef\optBeamerClassicFormat%
    \undef\optBeamerWideFormat%
    \geometry{a0paper, margin=8cm}% (2*sqrt(2)*sqrt(2)*sqrt(2)*sqrt(2))
    \setlength{\AMlayoutHeaderSize}{4cm}% (sqrt(2)*sqrt(2)*sqrt(2)*sqrt(2))
}%
\newcommand{\AMlayoutSetBeamerClassicFormat}{%
    \undef\optAfive%
    \undef\optAfour%
    \undef\optAthree%
    \undef\optAtwo%
    \undef\optAone%
    \undef\optAzero%
    \def\optBeamerClassicFormat{x}%
    \undef\optBeamerWideFormat%
    \geometry{paperwidth=25.4cm, paperheight=19.05cm, left=1cm, right=1cm}%
    \setlength{\AMlayoutHeaderSize}{1cm}%
}%
\newcommand{\AMlayoutSetBeamerWideFormat}{%
    \undef\optAfive%
    \undef\optAfour%
    \undef\optAthree%
    \undef\optAtwo%
    \undef\optAone%
    \undef\optAzero%
    \undef\optBeamerClassicFormat%
    \def\optBeamerWideFormat{x}%
    \geometry{paperwidth=25.4cm, paperheight=14.29cm, left=1cm, right=1cm}%
    \setlength{\AMlayoutHeaderSize}{1cm}%
}%
%
% Set chosen format
\ifdefined\optAfive\AMlayoutSetAfive\fi%
\ifdefined\optAfour\AMlayoutSetAfour\fi%
\ifdefined\optAthree\AMlayoutSetAthree\fi%
\ifdefined\optAtwo\AMlayoutSetAtwo\fi%
\ifdefined\optAone\AMlayoutSetAone\fi%
\ifdefined\optAzero\AMlayoutSetAzero\fi%
\ifdefined\optBeamerClassicFormat\AMlayoutSetBeamerClassicFormat\fi%
\ifdefined\optBeamerWideFormat\AMlayoutSetBeamerWideFormat\fi%
%
% Absolute positioning on page
% ----------------------------
\newcommand{\AMlayoutPutAtCenter}[2][0]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[remember picture, overlay]%
        \node[shift={($ (\paperwidth/2,\paperheight/2) $)},outer sep=0,inner sep=0,rotate=#1,align=left] at (current page.south west) {#2};%
    \end{tikzpicture}%
}%
\newcommand{\AMlayoutPutAtNorthWest}[4][0]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[remember picture, overlay]%
        \node[shift={(#2,-#3)},outer sep=0,inner sep=0,anchor = north west,rotate=#1,align=left] at (current page.north west) {#4};%
    \end{tikzpicture}%
}%
\newcommand{\AMlayoutPutAtNorthWestCentered}[4][0]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[remember picture, overlay]%
        \node[shift={(#2,-#3)},outer sep=0,inner sep=0,rotate=#1,align=left] at (current page.north west) {#4};%
    \end{tikzpicture}%
}%
\newcommand{\AMlayoutPutAtNorthEast}[4][0]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[remember picture, overlay]%
        \node[shift={(-#2,-#3)},outer sep=0,inner sep=0,anchor = north east,rotate=#1,align=left] at (current page.north east) {#4};%
    \end{tikzpicture}%
}%
\newcommand{\AMlayoutPutAtNorthEastCentered}[4][0]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[remember picture, overlay]%
        \node[shift={(-#2,-#3)},outer sep=0,inner sep=0,rotate=#1,align=left] at (current page.north east) {#4};%
    \end{tikzpicture}%
}%
\newcommand{\AMlayoutPutAtSouthWest}[4][0]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[remember picture, overlay]%
        \node[shift={(#2,#3)},outer sep=0,inner sep=0,anchor = south west,rotate=#1,align=left] at (current page.south west) {#4};%
    \end{tikzpicture}%
}%
\newcommand{\AMlayoutPutAtSouthWestCentered}[4][0]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[remember picture, overlay]%
        \node[shift={(#2,#3)},outer sep=0,inner sep=0,rotate=#1,align=left] at (current page.south west) {#4};%
    \end{tikzpicture}%
}%
\newcommand{\AMlayoutPutAtSouthEast}[4][0]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[remember picture, overlay]%
        \node[shift={(-#2,#3)},outer sep=0,inner sep=0,anchor = south east,rotate=#1,align=left] at (current page.south east) {#4};%
    \end{tikzpicture}%
}%
\newcommand{\AMlayoutPutAtSouthEastCentered}[4][0]{%
    \AMtikzExternalizeSkipNext%
    \begin{tikzpicture}[remember picture, overlay]%
        \node[shift={(-#2,#3)},outer sep=0,inner sep=0,rotate=#1,align=left] at (current page.south east) {#4};%
    \end{tikzpicture}%
}%
%
% Headers
% -------
\newcommand{\AMlayoutHeaderCDTUMLogoOnly}[1][TUMAnniGreen2]{%
    % Output TUM logo
    \AMlayoutPutAtNorthEast{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTUM{\AMlayoutHeaderSize}{TUMBlue}}%
    % Output anniversary badge
    \ifdefined\optExzellenz%
        \AMlayoutPutAtNorthEast{3.9\AMlayoutHeaderSize}{0cm}{\AMlogoAnniversary[\AMlayoutHeaderSize]{#1}}%
    \fi%
}%
\newcommand{\AMlayoutHeaderCDTUM}[1][TUMAnniGreen2]{%
    % Output TUM tercet
    \AMlayoutPutAtNorthWest{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTercetTUM{\AMlayoutHeaderSize}}%
    % Output TUM logo
    \AMlayoutPutAtNorthEast{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTUM{\AMlayoutHeaderSize}{TUMBlue}}%
    % Output anniversary badge
    \ifdefined\optExzellenz%
        \AMlayoutPutAtNorthEast{3.9\AMlayoutHeaderSize}{0cm}{\AMlogoAnniversary[\AMlayoutHeaderSize]{#1}}%
    \fi%
}%
\newcommand{\AMlayoutHeaderCDDepartment}[1][TUMAnniGreen2]{%
    % Output school logo
    \AMlayoutPutAtNorthWest{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoDepartment{\AMlayoutHeaderSize}{TUMBlue}}%
    % Output TUM tercet
    \AMlayoutPutAtNorthWest{2.7\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTercetSchool{\AMlayoutHeaderSize}}%
    % Output TUM logo
    \AMlayoutPutAtNorthEast{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTUM{\AMlayoutHeaderSize}{TUMBlue}}%
    % Output anniversary badge
    \ifdefined\optExzellenz%
        \AMlayoutPutAtNorthEast{3.9\AMlayoutHeaderSize}{0cm}{\AMlogoAnniversary[\AMlayoutHeaderSize]{#1}}%
    \fi%
}%
\newcommand{\AMlayoutHeaderCDSchoolTUMLogoOnly}{%
	% Output TUM tercet
	\AMlayoutPutAtNorthWest{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTercetSchool{\AMlayoutHeaderSize}}%
	% Output TUM logo
	\AMlayoutPutAtNorthEast{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTUM{\AMlayoutHeaderSize}{TUMBlue}}%
}%
\newcommand{\AMlayoutHeaderCDChair}[1][TUMAnniGreen2]{%
    % Output TUM tercet
    \AMlayoutPutAtNorthWest{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTercetChair{\AMlayoutHeaderSize}}%
    % Output TUM logo
    \AMlayoutPutAtNorthEast{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTUM{\AMlayoutHeaderSize}{TUMBlue}}%
    % Output anniversary badge
    \ifdefined\optExzellenz%
        \AMlayoutPutAtNorthEast{3.9\AMlayoutHeaderSize}{0cm}{\AMlogoAnniversary[\AMlayoutHeaderSize]{#1}}%
    \fi%
}%
%
% Debugging
% ---------
\newcommand{\AMlayoutCDBorder}{%
    \AMlayoutPutAtNorthWest{0cm}{0cm}{%
        \begin{tikzpicture}%
            \draw [red] (\AMlayoutHeaderSize,0cm) -- (\AMlayoutHeaderSize,\paperheight);%
            \draw [red] (\paperwidth-\AMlayoutHeaderSize,0cm) -- (\paperwidth-\AMlayoutHeaderSize,\paperheight);%
            \draw [red] (0cm,\AMlayoutHeaderSize) -- (\paperwidth,\AMlayoutHeaderSize);%
            \draw [red] (0cm,\paperheight-\AMlayoutHeaderSize) -- (\paperwidth,\paperheight-\AMlayoutHeaderSize);%
            \draw [red] (0cm,\paperheight-2\AMlayoutHeaderSize) -- (\paperwidth,\paperheight-2\AMlayoutHeaderSize);%
            \draw [red] (0cm,\paperheight-3\AMlayoutHeaderSize) -- (\paperwidth,\paperheight-3\AMlayoutHeaderSize);%
            \draw [red] (\paperwidth-2.9\AMlayoutHeaderSize,\paperheight-3\AMlayoutHeaderSize) -- (\paperwidth-2.9\AMlayoutHeaderSize,\paperheight);%
            \draw [red] (\paperwidth-3.9\AMlayoutHeaderSize,\paperheight-3\AMlayoutHeaderSize) -- (\paperwidth-3.9\AMlayoutHeaderSize,\paperheight);%
        \end{tikzpicture}%
    }%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMlayout.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Commands ---
%================
% Headers
% -------
% Custom design (chair)
\newcommand{\AMlayoutHeaderCustomChair}[1][TUMAnniGreen2]{%
    % Output AM logo
    \AMlayoutPutAtNorthWest{\AMlayoutHeaderSize}{0.85\AMlayoutHeaderSize}{\AMlogoAM{1.15\AMlayoutHeaderSize}{AMLogoBlue1}{AMLogoBlue2}}%
    % Output TUM tercet
    \AMlayoutPutAtNorthWest{2.2\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTercetChair{\AMlayoutHeaderSize}}%
    % Output TUM logo
    \AMlayoutPutAtNorthEast{\AMlayoutHeaderSize}{\AMlayoutHeaderSize}{\AMlogoTUM{\AMlayoutHeaderSize}{TUMBlue}}%
    % Output anniversary badge
    \ifdefined\optExzellenz%
        \AMlayoutPutAtNorthEast{3.9\AMlayoutHeaderSize}{0cm}{\AMlogoAnniversary[\AMlayoutHeaderSize]{#1}}%
    \fi%
}%
%
%
% Title pages
% -----------
\newcommand{\AMlayoutTitlePageDefault}[3]{%
    \AMlayoutHeaderCustomChair%
    \vskip1.1cm\noindent%
    \ifx\\#1\\\else%    %check, if argument is not empty
        \textbf{\huge #1}%
    \fi%
    \ifx\\#2\\\else%    %check, if argument is not empty
        \\[1em]\textbf{\Large \color{TUMBlue} #2}%
    \fi%
    \ifx\\#3\\\else%    %check, if argument is not empty
        \\[1em]#3%
    \fi%
    \par%
}%
%
%
%
\else\fi% End-check if module AMlayout is enabled
%###################################################################################################
%--- END MODULE AMlayout ---------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMbiblio -------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMbiblio% Start-check if module AMbiblio is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMbiblio.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%=======================
%--- Option handling ---
%=======================
% Check, if a backend has been specified
\ifdefined\optBiber%
    \def\BackendSpecified{x}%
\else\fi%
\ifdefined\optBibtex%
    \def\BackendSpecified{x}%
\else\fi%
%
% Use numeric citation style as default
\ifdefined\optBibstyleNumeric\else%
\ifdefined\optBibstyleAlphabetic\else%
\ifdefined\optBibstyleAuthorYear\else%
    \def\optBibstyleNumeric{x}%
\fi%
\fi%
\fi%
%
%
% Perform rest only, if a backend has been specified
\ifdefined\BackendSpecified%
%================
%--- Packages ---
%================
% Select backend
% --------------
\ifdefined\optBiber% Load biblatex package with biber as backend
    \PassOptionsToPackage{backend=biber,defernumbers=true}{biblatex}%
\else\fi%
\ifdefined\optBibtex% Load biblatex package with bibtex as backend
    \PassOptionsToPackage{backend=bibtex}{biblatex}%
\else\fi%
%
% Select citation style
% ---------------------
\ifdefined\optBibstyleNumeric%
    \PassOptionsToPackage{style=numeric-comp}{biblatex}%
\else\fi%
\ifdefined\optBibstyleAlphabetic%
    \PassOptionsToPackage{style=alphabetic}{biblatex}%
\else\fi%
\ifdefined\optBibstyleAuthorYear%
    \PassOptionsToPackage{style=authoryear-comp}{biblatex}%
\else\fi%
%
% Load package
% ------------
\RequirePackage{biblatex}%
%
% Set options
% -----------
% Abbreviate first name with inits
\@ifpackagelater{biblatex}{2016/03/13}%
{%
    \ExecuteBibliographyOptions{giveninits=true}% new option
}%
{%
    \ExecuteBibliographyOptions{firstinits=true}% deprecated option
}%
%
% Sorting
% -------
\ExecuteBibliographyOptions{sorting=nyt}%  Set sorting order
\ExecuteBibliographyOptions{maxnames=99}%  Set maximum names of a reference
\DeclareNameAlias{sortname}{family-given}% Sort by last name
\DeclareNameAlias{default}{family-given}%  Print last name first
%
%
% End test, if a backend has been specified
\else\fi%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMbiblio.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if module AMbiblio is enabled
%###################################################################################################
%--- END MODULE AMbiblio ---------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMnames --------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMnames% Start-check if module AMnames is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMnames.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%==================
%--- Professors ---
%==================
% President
% ---------
\def\AMnamesProfHerrmann{Prof.\ Dr.\ Dr.\ h.c.\ mult.\ Wolfgang A. Herrmann}%
\def\AMnamesProfHofmann{Prof.\ Dr.\ Thomas Hofmann}%
\def\AMnamesPresident{\AMnamesProfHofmann}%
%
% Active professors
% -----------------
% School of Mechanical Engineering
\def\AMnamesProfAdams{Prof.\ Dr.-Ing.\ Nikolaus A. Adams}%
\def\AMnamesProfBengler{Prof.\ Dr.\ phil.\ Klaus Bengler}%
\def\AMnamesProfBerensmeier{Prof.\ Dr.\ Sonja Berensmeier}%
\def\AMnamesProfBottasso{Prof.\ Dr.\ Carlo L. Bottasso}%
\def\AMnamesProfCaccamo{Prof.\ Dr.\ Marco Caccamo}%
\def\AMnamesProfDaub{Prof.\ Dr.\ Rüdiger Daub}%
\def\AMnamesProfDrechsler{Prof.\ Dr.-Ing.\ Klaus Drechsler}%
\def\AMnamesProfFottner{Prof.\ Dr.-Ing.\ Johannes Fottner}%
\def\AMnamesProfGee{Prof.\ Dr.-Ing.\ Michael W. Gee}%
\def\AMnamesProfGrosse{Prof.\ Dr.-Ing.\ habil.\ Dipl.-Geophys.\ Christian Große}%
\def\AMnamesProfGuemmer{Prof.\ Dr.-Ing.\ Volker Gümmer}%
\def\AMnamesProfGuenthner{Prof.\ Dr.-Ing.\ Willibald A. Günthner}%
\def\AMnamesProfHaidn{Prof.\ Dr.-Ing.\ Oskar J. Haidn}%
\def\AMnamesProfHajek{Prof.\ Dr.-Ing.\ Manfred Hajek}%
\def\AMnamesProfHolzapfel{Prof.\ Dr.-Ing.\ Florian Holzapfel}%
\def\AMnamesProfHornung{Prof.\ Dr.-Ing.\ Mirko Hornung}%
\def\AMnamesProfKaltenbach{Prof.\ Dr.-Ing.\ Hans-Jakob Kaltenbach}%
\def\AMnamesProfKlein{Prof.\ Dr.-Ing.\ Harald Klein}%
\def\AMnamesProfKoutsourelakis{Prof.\ Phaedon-Stelios Koutsourelakis, Ph.D.}%
\def\AMnamesProfKremling{Prof.\ Dr.-Ing.\ Andreas Kremling}%
\def\AMnamesProfLieleg{Prof.\ Dr.\ Oliver Lieleg}%
\def\AMnamesProfLienkamp{Prof.\ Dr.-Ing.\ Markus Lienkamp}%
\def\AMnamesProfLindemann{Prof.\ Dr.-Ing.\ Udo Lindemann}%
\def\AMnamesProfLohmann{Prof.\ Dr.-Ing.\ habil.\ Boris Lohmann}%
\def\AMnamesProfLueth{Prof.\ Dr.\ rer.\ nat.\ Tim C. Lüth}%
\def\AMnamesProfMacianJuan{Prof.\ Rafael Macián-Juan, Ph.D.}%
\def\AMnamesProfMarburg{Prof.\ Dr.-Ing.\ Steffen Marburg}%
\def\AMnamesProfNeu{Prof.\ Dr.\ Rudolf Neu}%
\def\AMnamesProfPolifke{Prof.\ Wolfgang Polifke, Ph.D.}%
\def\AMnamesProfProvost{Prof.\ Dr.\ Julien Provost}%
\def\AMnamesProfReinhart{Prof.\ Dr.-Ing.\ Gunther Reinhart}%
\def\AMnamesProfRixen{Prof.\ dr.ir.\ Daniel J. Rixen}%
\def\AMnamesProfSattelmayer{Prof.\ Dr.-Ing.\ Thomas F. Sattelmayer}%
\def\AMnamesProfSenner{Prof.\ Dr.-Ing.\ Veit Senner}%
\def\AMnamesProfSpannerUlmer{Prof.\ Dr.\ phil.\ habil.\ Dr.-Ing.\ Birgit Spanner-Ulmer}%
\def\AMnamesProfSpliethoff{Prof.\ Dr.-Ing.\ Hartmut Spliethoff}%
\def\AMnamesProfStahl{Prof.\ Dr.-Ing.\ Karsten Stahl}%
\def\AMnamesProfVogelHeuser{Prof.\ Dr.-Ing.\ Birgit Vogel-Heuser}%
\def\AMnamesProfVolk{Prof.\ Dr.-Ing.\ Wolfram Volk}%
\def\AMnamesProfWachtmeister{Prof.\ Dr.\ Georg Wachtmeister}%
\def\AMnamesProfWall{Prof.\ Dr.-Ing.\ Wolfgang A. Wall}%
\def\AMnamesProfWalter{Prof.\ Dr.\ Ulrich Walter}%
\def\AMnamesProfWerner{Prof.\ Dr.\ Ewald Werner}%
\def\AMnamesProfWeusterBotz{Prof.\ Dr.-Ing.\ Dirk Weuster-Botz}%
\def\AMnamesProfZaeh{Prof.\ Dr.-Ing.\ Michael Zäh}%
\def\AMnamesProfZimmermann{Prof.\ Dr.\ Markus Zimmermann}%
%
% School of Informatics
\def\AMnamesProfAlthoff{Prof.\ Dr.-Ing.\ Matthias Althoff}%
\def\AMnamesProfBurschka{Prof.\ Dr.-Ing.\ Darius Burschka}%
\def\AMnamesProfKnoll{Prof.\ Dr.-Ing.\ habil.\ Alois C. Knoll}%
\def\AMnamesProfPretschner{Prof.\ Dr.\ Alexander Pretschner}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMnames.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%==================
%--- Professors ---
%==================
% Retired professors
% ------------------
\def\AMnamesProfUlbrich{Univ.-Prof.\ Dr.-Ing.\ Dr.-Ing.\ habil.\ Heinz Ulbrich,\ i.R.}%
\def\AMnamesProfPfeiffer{Univ.-Prof.\ Dr.-Ing., Dr.-Ing.\,E.h., Dr.-Ing.\,h.c.\,mult.\ Friedrich Pfeiffer, i.R.}%
%
%
%=================
%--- Employees ---
%=================
% Academic director
% -----------------
\def\AMnamesStaffThuemmel{PD\ Dr.-Ing.\ habil.\ Thomas Thümmel}%
%
%
%
\else\fi% End-check if module AMnames is enabled
%###################################################################################################
%--- END MODULE AMnames ----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMutils --------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMutils% Start-check if module AMutils is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMutils.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{excludeonly}%              Provides command to exclude source files (for editing large documents)
\RequirePackage{epstopdf}%                 Automatically converts .eps files to .pdf
\RequirePackage{xspace}%                   Provides command to insert/skip whitespace after a command
\RequirePackage{cancel}%                   Provides commands to strike through text and math expressions
\RequirePackage{datetime}%                 Provides current date and time
\RequirePackage{eurosym}%                  Provides the euro symbol
\RequirePackage{blindtext}%                Helper to write blind text for templates
\ifdefined\optAMbeamer\else%              Avoid clash with beamer baseclass
    \RequirePackage[clearempty]{titlesec}% Customize captions etc. (has to be leaded BEFORE hyperref)
    \RequirePackage{titletoc}%             Customize table of contents (has to be leaded BEFORE hyperref)
    \RequirePackage{pdfcomment}%           Allows PDF-comments and PDF-annotations (loads package hyperref -> should be the last package!)
\fi%
%
%
%================
%--- Commands ---
%================
\DeclareExpandableDocumentCommand{\AMutilsTodo}{O{red}m}{%
    \ifdefined\optHideTodos\else%
        {\texorpdfstring{\color{#1}\textbf{ToDo:} #2}{}}%
    \fi%
}%
\DeclareExpandableDocumentCommand{\AMutilsTodoW}{O{red}m}{%
    \ifdefined\optHideTodos\else%
        \AMutilsTodo[#1]{#2}\AMcoreLogWarning{#2}%
    \fi%
}%
\newcommand{\AMutilsTodoSpacer}[2][]%
{%
    \ifdefined\optHideTodos\else%
        \ifx\\#1\\% check, if first argument is not empty
            \ \\[-0.5\baselineskip]%
        \else%
            \ \\{\color{red}\footnotesize ToDo: #1}\ \\[-0.8\baselineskip]%
        \fi%
        \tikz \draw[red,line width=0.5mm] (0cm,0cm) -- (\textwidth-1.4pt,0cm);%
        \AMcoreRepetition{#2}{\\[-\baselineskip]\vspace{1mm}\ \\[-\baselineskip]\tikz \draw[red,line width=0.5mm] (0cm,0cm) -- (0cm,1mm); \hfill \tikz \draw[red,line width=0.5mm] (0cm,0cm) -- (0cm,1mm);}%
        \\[-\baselineskip]\tikz \draw[red,line width=0.5mm] (0cm,0cm) -- (\textwidth-1.4pt,0cm);\\%
    \fi%
}%
%
% PDF comments, annotations and highlights
% ----------------------------------------
\DeclareExpandableDocumentCommand{\AMutilsPDFanno}{O{unknown}m}{%
    \ifdefined\optHideAnnotations\else%
        \texorpdfstring{\pdfcomment[icon=Note,color=yellow,opacity=0.5,author={#1}]{#2}}{}%
    \fi%
}%
\DeclareExpandableDocumentCommand{\AMutilsPDFannoColor}{O{unknown}mm}{%
    \ifdefined\optHideAnnotations\else%
        \texorpdfstring{\pdfcomment[icon=Note,color=#3,opacity=0.5,author={#1}]{#2}}{}%
    \fi%
}%
\DeclareExpandableDocumentCommand{\AMutilsPDFhigh}{O{yellow}m}{%
    \ifdefined\optHideAnnotations\else%
        \texorpdfstring{\pdfmarkupcomment[markup=Highlight,color=#1]{#2}{}}{}%
    \fi%
}%
%
% Date and time
% -------------
\newcommand{\AMutilsDate}[3]{\AMlangGerEng{#1. \AMlangMonth{#2} #3}{\AMlangMonth{#2} #1, #3}}
%
\newcommand{\AMutilsToday}{\AMutilsDate{\the\day}{\the\month}{\the\year}}
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMutils.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if module AMutils is enabled
%###################################################################################################
%--- END MODULE AMutils ----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN MODULE AMref ----------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMref% Start-check if module AMref is enabled
%
% >>> CONTENTS OF FILE source/TUM/Modules/TUMref.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
% Pre-load packages before loading hyperref (trick to be able to load packages before hyperref without modifying this package)
\IfFileExists{pre_hyperref_packages.tex}{\input{pre_hyperref_packages.tex}}{}%
%
\RequirePackage{hyperref}%
\ifdefined\optBlackRefs%
    \hypersetup{%
        %bookmarks=true,%
        linkcolor=TUMBlack,%
        citecolor=TUMBlack,%
        urlcolor=TUMBlack,%
        colorlinks=true}%
\else%
    \hypersetup{%
        %bookmarks=true,%
        linkcolor=TUMBlue,%
        citecolor=TUMBlue,%
        urlcolor=TUMBlue,%
        colorlinks=true}%
\fi%
%
\ifdefined\optAMbeamer\else%              Avoid clash with beamer baseclass
    \ifdefined\optGerman%
        \PassOptionsToPackage{ngerman}{cleveref}%
    \else\fi%
    \ifdefined\optEnglish%
        \PassOptionsToPackage{english}{cleveref}%
    \else\fi%
    \RequirePackage{cleveref}%
\fi%
%
%================
%--- Commands ---
%================
% Set PDF meta data
\newcommand{\AMrefSetPDFMetadata}[6]{%
    \hypersetup{pdftitle={#1},pdfsubject={#2},pdfauthor={#3},pdfkeywords={#4},pdfcreator={#5},pdfproducer={#6}}%
}%
%
%
%========================
%--- Execute defaults ---
%========================
\AMrefSetPDFMetadata{}{}{}{}{TUMlatex project}{}%
\urlstyle{same}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Modules/AMref.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if module AMref is enabled
%###################################################################################################
%--- END MODULE AMref ------------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMstudentexpose -------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMstudentexpose% Start-check if style AMstudentexpose is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMstudentexpose.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%
\RequirePackage{fancyhdr}%
%
%=============
%--- Style ---
%=============
% Set default layout
\geometry{left=3cm,right=3cm,top=2.5cm,bottom=3cm,headheight=14pt}%
%
% Paragraph indentation
\AMlayoutSetIndent%
%
% Package titlesec: section style
\titlespacing*\section{0pt}{30pt plus 4pt minus 2pt}{16pt plus 2pt minus 2pt}%
\titlespacing*\subsection{0pt}{20pt plus 4pt minus 2pt}{12pt plus 2pt minus 2pt}%
\titlespacing*\subsubsection{0pt}{10pt plus 4pt minus 2pt}{2pt plus 2pt minus 2pt}%
\titleformat{\section}{\color{TUMBlack}\sffamily\Large\bfseries}{\thesection}{1em}{}%
\titleformat{\subsection}{\color{TUMBlack}\sffamily\large\bfseries}{\thesubsection}{1em}{}%
\titleformat{\subsubsection}{\color{TUMBlack}\sffamily\normalsize\bfseries}{\thesubsubsection}{1em}{}%
%
% Package fancyhdr: header and footer style
\def\AMfooterAuthor{Author}%
\def\AMfooterDate{Date}%
\fancypagestyle{AMstudentexposep}{%
	\AtBeginDocument{%
		\renewcommand{\sectionmark}[1]{\markright{\thesection\ \ ##1}}%
	}%
	\fancyhf{}%
	\fancyhead[L,R]{}%
	\fancyfoot[L]{\textit{\footnotesize\AMfooterAuthor, \AMlangChair, \AMfooterDate}}%
	\fancyfoot[R]{\textit{\thepage}}%
	\renewcommand{\headrulewidth}{0pt}%
	\renewcommand{\footrulewidth}{0pt}%
}%
%
% Set pagestyle
\pagestyle{AMstudentexposep}%
%
% Change plain style
\fancypagestyle{plain}{%
	\fancyhf{} % clear all header and footer fields
	\fancyfoot[C]{\normalcolor\large\bfseries\sffamily\thepage}%
	\renewcommand{\headrulewidth}{0pt}%
	\renewcommand{\footrulewidth}{0pt}%
}%
%
% Caption style
\captionsetup{labelfont=bf,font=small}%
%
%
%=================
%--- Titlepage ---
%=================
\newcommand{\AMstudentexposeTitlePage}[7]{%
    \AMstudentexposeHeader{#1}{#2}{#3}{#4}{#5}{#6}{#7}%
    \AMstudentexposeReview%
}%
\newcommand{\AMstudentexposeHeader}[7]{%
    \def\AMfooterAuthor{#3}%
	\def\AMfooterDate{#7}%
    \AMrefSetPDFMetadata{Expos\'{e} #1}{}{#3}{}{}{}%
    \AMlayoutHeaderCDTUM%
    \addcontentsline{toc}{section}{\AMlangGerEng{Titelblatt}{Titlepage}}%
    %
    % Print title and student information
    \par\vspace*{0.5cm}%
    \begingroup%
        \noindent\huge \color{TUMBlue} \bfseries Expos\'{e} #1\\[0.2cm]%
    \endgroup%
    \begingroup%
        \large\noindent%
        \begin{tabular}{@{}lp{0.84\linewidth}@{}}% %Ensures linebreak for long titles
            Name: & #3\\%
            Matr. Nr.: & #4\\%
            EMail: & #5\\%
            \AMlangSupervisor: & #6\\%
            \AMlangGerEng{Institut}{Institute}: & \AMlangChair\\%
            \AMlangGerEng{Titel}{Title}: & #2\\%
            \large\AMlangDate: & #7\\%
        \end{tabular}\par%
    \endgroup%
    \vspace*{0.2cm}%
    \noindent\rule{\textwidth}{0.7pt}%
}%
\newcommand{\AMstudentexposeTitlePageBachelorsThesis}[6]{\AMstudentexposeTitlePage{\AMlangGerEng{zur}{on} \AMlangBachelorsThesis}{#1}{#2}{#3}{#4}{#5}{#6}}%
\newcommand{\AMstudentexposeTitlePageSemesterThesis}[6]{\AMstudentexposeTitlePage{\AMlangGerEng{zur}{on} \AMlangSemesterThesis}{#1}{#2}{#3}{#4}{#5}{#6}}%
\newcommand{\AMstudentexposeTitlePageMastersThesis}[6]{\AMstudentexposeTitlePage{\AMlangGerEng{zur}{on} \AMlangMastersThesis}{#1}{#2}{#3}{#4}{#5}{#6}}%
\newcommand{\AMstudentexposeTitlePageIDP}[6]{\AMstudentexposeTitlePage{\AMlangGerEng{zum}{on} \AMlangInterdisciplinaryProject}{#1}{#2}{#3}{#4}{#5}{#6}}%
%
%==================
%--- Backmatter ---
%==================
\newcommand{\AMstudentexposeReview}{%
    \AtEndDocument{%
        \clearpage%
        \section*{\AMlangGerEng{Überprüfung}{Review}}%
        \begingroup%
            \sffamily%
            \textit{\AMlangGerEng{Inhaltliche Ausführung erstellt von}{Content presentation created by}:}\\%
            \begin{tabular}{@{}p{4.7cm}p{4.7cm}p{4.7cm}@{}}%
                 & & \\[3mm]%
                \rule{2.5cm}{0.3mm}\;, \;\rule{1.7cm}{0.3mm} & \rule{2.5cm}{0.3mm}\;, \;\rule{1.7cm}{0.3mm} & \rule{2.5cm}{0.3mm}\;, \;\rule{1.7cm}{0.3mm}\\%
                \textit{\AMlangGerEng{Ort, Datum}{Place, date}} & \textit{\AMlangGerEng{Ort, Datum}{Place, date}} & \textit{\AMlangGerEng{Ort, Datum}{Place, date}}\\[8mm]%
                \rule{4.7cm}{0.3mm} & \rule{4.7cm}{0.3mm} & \rule{4.7cm}{0.3mm}\\%
                \AMlangGerEng{Student/in}{Student} & \AMlangGerEng{Erstprüfer/in,}{First examiner,} & \AMlangGerEng{Zweitprüfer/in,}{Second examiner,}\\%
                & \AMlangGerEng{Betreuender wiss. Mitarbeiter am Lehrstuhl}{Supervising scientific staff at the chair} & \AMlangGerEng{Zentrum für Schlüsselkompetenzen}{Center of Key Competencies}\\%
            \end{tabular}%
        \endgroup%
    }%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMstudentexpose.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if style AMstudentexpose is enabled
%###################################################################################################
%--- END STYLE AMstudentexpose ---------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMstudentthesiscover --------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMstudentthesiscover% Start-check if style AMstudentthesiscover is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMstudentthesiscover.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{calc}%
%
%
%=============
%--- Style ---
%=============
% Set page layout
\geometry{paperwidth=450mm, paperheight=320mm, margin={0pt,0pt}}%
\thispagestyle{empty}%
%
%
%=====================
%--- Configuration ---
%=====================
\newlength{\CoverMarkLength}%
\setlength{\CoverMarkLength}{3mm}%
%
%
%=============
%--- Cover ---
%=============
\newcommand{\AMstudentthesisCover}[4][]{%
	% Get page count of linked PDF
	\ifx\\#1\\%
		\pdfximage{#4}%
	    \def\CoverPDFpageCount{\pdflastximagepages}%
	\else%
		\def\CoverPDFpageCount{#1}%
    \fi%
	%
	% Compute spine height
	\newlength{\SpineHeight}%
	\setlength{\SpineHeight}{\CoverPDFpageCount mm}%
	\setlength{\SpineHeight}{0.05\SpineHeight}% page-count/2 * 0.1mm per page
	\ifdim\SpineHeight<3mm%
	    \AMcoreLogWarning{Too few pages - spine height is too low for printing cover! Consider printing one-sided or filling with empty pages!}%
	\else\fi%
	\setlength{\SpineHeight}{\maxof{\SpineHeight}{3mm}}% minimum-height: 3mm
	%
	% Draw content
	\begin{tikzpicture}%
		% Background
		\fill [TUMWhite] (0mm,0mm) rectangle (210mm+\SpineHeight+210mm,297mm);%
	    %
	   	% Draw borders
	   	\ifdefined\optCoverWithBorders%
	   		\draw [red] (0mm,0mm) rectangle (210mm,297mm);%
	   		\draw [red] (210mm,0mm) rectangle (210mm+\SpineHeight,297mm);%
	   		\draw [red] (210mm+\SpineHeight,0mm) rectangle (210mm+\SpineHeight+210mm,297mm);%
	   	\else\fi%
	    %
	    % Draw marks
	   	\ifdefined\optCoverWithMarks%
	        % Left marks
	   		\draw [TUMBlack] (-\CoverMarkLength,0mm) -- (0mm,0mm);%
	   		\draw [TUMBlack] (-\CoverMarkLength,297mm) -- (0mm,297mm);%
	        % Right marks
	   		\draw [TUMBlack] (210mm+\SpineHeight+210mm,0mm) -- (210mm+\SpineHeight+210mm+\CoverMarkLength,0mm);%
	   		\draw [TUMBlack] (210mm+\SpineHeight+210mm,297mm) -- (210mm+\SpineHeight+210mm+\CoverMarkLength,297mm);%
	        % Bottom marks
	  		\draw [TUMBlack] (0mm,-\CoverMarkLength) -- (0mm,0mm);%
	   		\draw [TUMBlack] (210mm,-\CoverMarkLength) -- (210mm,0mm);%
	   		\draw [TUMBlack] (210mm+\SpineHeight,-\CoverMarkLength) -- (210mm+\SpineHeight,0mm);%
	   		\draw [TUMBlack] (210mm+\SpineHeight+210mm,-\CoverMarkLength) -- (210mm+\SpineHeight+210mm,0mm);%
	        % Bottom marks
	  		\draw [TUMBlack] (0mm,297mm) -- (0mm,297mm+\CoverMarkLength);%
	   		\draw [TUMBlack] (210mm,297mm) -- (210mm,297mm+\CoverMarkLength);%
	   		\draw [TUMBlack] (210mm+\SpineHeight,297mm) -- (210mm+\SpineHeight,297mm+\CoverMarkLength);%
	   		\draw [TUMBlack] (210mm+\SpineHeight+210mm,297mm) -- (210mm+\SpineHeight+210mm,297mm+\CoverMarkLength);%
	   	\else\fi%
	    %
	    % Draw first page of pdf
	    \node[inner sep=0pt, outer sep=0pt, anchor=south west] at (210mm+\SpineHeight,0mm) {\includegraphics[page=1, width=210mm]{#4}};%
	    %
	    % Draw spine
	    \node[inner sep=0pt, outer sep=0pt, anchor=west, rotate=90] at (210mm+0.5\SpineHeight,30mm) {\bfseries\fontsize{7}{7}\selectfont #2};%
	    \node[inner sep=0pt, outer sep=0pt, anchor=west, rotate=90] at (210mm+0.5\SpineHeight,80mm) {\bfseries\fontsize{7}{7}\selectfont #3};%
	\end{tikzpicture}%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMstudentthesiscover.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if style AMstudentthesiscover is enabled
%###################################################################################################
%--- END STYLE AMstudentthesiscover ----------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMbulletin ------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMbulletin% Start-check if style AMbulletin is enabled
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMbulletin.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%===============================
%--- Default Option handling ---
%===============================
% Select student-bulletin type, if no option has been passed
\ifdefined\optStudentBulletin\else%
\ifdefined\optResearchBulletin\else%
\ifdefined\optTeachingBulletin\else%
    \def\optStudentBulletin{x}%
\fi%
\fi%
\fi%
%
%
%=============
%--- Style ---
%=============
% Set bulletin formatting
\ifdefined\optStudentBulletin%
    \AMfontSetDocumentFontSizeXII%      Font-size = 12
    \def\BulletinWidth{170mm}%          Width of bulletin
    \def\BulletinHeight{125mm}%         Height of bulletin
\fi%
\ifdefined\optResearchBulletin%
    \AMfontSetDocumentFontSizeXVI%      Font-size = 16
    \def\BulletinWidth{235mm}%          Width of bulletin
    \def\BulletinHeight{150mm}%         Height of bulletin
\fi%
\ifdefined\optTeachingBulletin%
    \AMfontSetDocumentFontSizeXVI%      Font-size = 16
    \def\BulletinWidth{235mm}%          Width of bulletin
    \def\BulletinHeight{150mm}%         Height of bulletin
\fi%
\normalsize%
%
% Page layout
\ifdefined\optBulletinCropPage%
    \geometry{paperwidth=\BulletinWidth,paperheight=\BulletinHeight,left=0pt,right=0pt,top=0pt,bottom=0pt}%
\else%
    \geometry{a4paper,landscape,margin=2cm}%
\fi%
%
% Paragraph indentation
\AMlayoutSetIndent%
%
%
%================
%--- Bulletin ---
%================
\newcommand{\AMbulletinFrame}{%
    \thispagestyle{empty}%
    % Draw frame
    \ifdefined\optBulletinCropPage%
        \fill[TUMWhite] (0,0) rectangle (\BulletinWidth,\BulletinHeight);%
    \else%
        \draw[black,line width = 1pt] (0,0) rectangle (\BulletinWidth,\BulletinHeight);%
    \fi%
    % Draw AM logo
    \node[inner sep=0pt,anchor=north west] at (7mm,\BulletinHeight-5.5mm) {\AMlogoAM{1.15cm}{AMLogoBlue1}{AMLogoBlue2}};%
    % Draw TUM tercet
    \node[inner sep=0pt,anchor=north west] at (19mm,\BulletinHeight-7mm) {\AMlogoTercetChair{1cm}};%
    % Draw TUM logo
    \node[inner sep=0pt,anchor=north east] at (\BulletinWidth-7mm,\BulletinHeight-7mm) {\AMlogoTUM{1cm}{TUMBlue}};%
    % Draw separator line
    \draw[TUMBlue,line width = 1pt] (7mm,\BulletinHeight-2cm) -- (\BulletinWidth-7mm,\BulletinHeight-2cm);%
}%
%
\newcommand{\AMbulletinStudent}[7]{%
    \begin{tikzpicture}%
        % Draw frame
        \AMbulletinFrame%
        %
        % Draw text and optional image
        \node[inner sep=0pt,anchor=north west] at (7mm,\BulletinHeight-25mm) {%
            \parbox{\BulletinWidth-14mm}{%
                % Draw title
                {\huge \color{TUMBlue}\bfseries #1}\par\vspace*{0.7em}%
                % Draw type
                {\Large \color{TUMBlue4} #2}\par\vspace*{0.7em}%
                % Check, if image is given
                \ifstrempty{#7}{%
                    % ...without image -> only draw description
                    \parbox{\BulletinWidth-14mm}{#3}%
                }{%
                    % ...with image -> draw description and image (side by side)
                    \begin{tabular}[t]{@{}p{\BulletinWidth-67mm}l} #3 & \raisebox{-\height+9pt}{\resizebox{5cm}{!}{#7}} \end{tabular}%
                }%
            }%
        };%
        %
        % Draw topics and contact information
        \node[inner sep=0pt,anchor=south west] at (7mm,5mm) {%
            \parbox{\BulletinWidth-14mm}{%
                \begin{tabular}[t]{@{}ll}%
                    % Draw topics
                    \ifx\\#4\\\else%    %check, if argument is not empty
                        \bfseries\AMlangGerEng{Themen}{Topics} & #4\\%
                    \fi%
                    % Draw contact information
                    \rule{0pt}{1.5em}\bfseries\AMlangGerEng{Betreuer}{Advisor} & #5\\%
                    & #6%
                \end{tabular}%
            }%
        };%
    \end{tikzpicture}\par%
}%
%
\newcommand{\AMbulletinResearch}[7]{%
    \begin{tikzpicture}%
        % Draw frame
        \AMbulletinFrame%
        %
        % Draw text and optional image
        \node[inner sep=0pt,anchor=north west] at (7mm,\BulletinHeight-25mm) {%
            \parbox{\BulletinWidth-14mm}{%
                % Draw title
                {\huge \color{TUMBlue}\bfseries #1}\par\vspace*{0.7em}%
                % Draw area of research
                {\Large \color{TUMBlue4} #2}\par\vspace*{0.7em}%
                % Check, if image is given
                \ifstrempty{#6}{%
                    % ...without image -> only draw description
                    \parbox{\BulletinWidth-14mm}{#3}%
                }{%
                    % ...with image -> draw description and image (side by side)
                    \begin{tabular}[t]{@{}p{\BulletinWidth-67mm}l} #3 & \raisebox{-\height+11pt}{\resizebox{5cm}{!}{#6}} \end{tabular}%
                }%
            }%
        };%
        %
        % Draw foto
        \node[inner sep=0pt,anchor=south west] at (7mm,5mm) {\resizebox{2.4cm}{3.2cm}{#7}};%
        % Draw topics and contact information
        \node[inner sep=0pt,anchor=south west] at (34mm,5mm) {%
            \parbox{\BulletinWidth-14mm}{%
                \begin{tabular}[t]{@{}p{7cm}p{12cm}}%
                    \bfseries\AMlangGerEng{Kontakt}{Contact} & \bfseries\AMlangGerEng{Themen}{Topics}\\%
                    #5 & #4%
                \end{tabular}%
            }%
        };%
    \end{tikzpicture}\par%
}%
%
\newcommand{\AMbulletinTeaching}[7]{%
    \begin{tikzpicture}%
        % Draw frame
        \AMbulletinFrame%
        %
        % Draw text and optional image
        \node[inner sep=0pt,anchor=north west] at (7mm,\BulletinHeight-25mm) {%
            \parbox{\BulletinWidth-14mm}{%
                % Draw title
                {\huge \color{TUMBlue}\bfseries #1}\par\vspace*{0.7em}%
                % Draw tpye
                {\Large \color{TUMBlue4} #2}\par\vspace*{0.7em}%
                % Check, if image is given
                \ifstrempty{#7}{%
                    % ...without image -> only draw description
                    \parbox{\BulletinWidth-14mm}{#3}%
                }{%
                    % ...with image -> draw description and image (side by side)
                    \begin{tabular}[t]{@{}p{\BulletinWidth-92mm}l} #3 & \raisebox{-\height+11pt}{\resizebox{7.5cm}{!}{#7}} \end{tabular}%
                }%
            }%
        };%
        %
        % Draw topics and contact information
        \node[inner sep=0pt,anchor=south west] at (7mm,5mm) {%
            \parbox{\BulletinWidth-14mm}{%
                \begin{tabular}[t]{@{}ll}%
                    % Draw topics
                    \ifx\\#4\\\else%    %check, if argument is not empty
                        \bfseries\AMlangGerEng{Themen}{Topics} & #4\\%
                    \fi%
                    % Draw contact information
                    \rule{0pt}{1.5em}\bfseries\AMlangGerEng{Dozent}{Lecturer} & #5\\%
                    \bfseries\AMlangGerEng{Betreuer}{Assistant} & #6%
                \end{tabular}%
            }%
        };%
    \end{tikzpicture}\par%
}%
%
%
%
\else\fi% End-check if style AMbulletin is enabled
%###################################################################################################
%--- END STYLE AMbulletin --------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMexercise ------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMexercise% Start-check if style AMexercise is enabled
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMexercise.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%  Setup of section captions
\RequirePackage{fancyhdr}% Setup of header and footer
\RequirePackage{enumitem}% Enumerations
\RequirePackage{setspace}% Allows to change line spacing
%
%
%===================
%--- Definitions ---
%===================
\newcommand{\subject}{\textbf{MODULNAME}}%
\newcommand{\term}{\textbf{TERM}}%
%
%
%=============
%--- Style ---
%=============
% Set default layout
\geometry{left=2.5cm,right=1.5cm,top=2cm,bottom=2cm}%
%
% Set font style
\AMfontSetDocumentFontSizeX%
\setlength\headheight{28pt}%
\setlength\headsep{0.3cm}%
\onehalfspacing%
%
% Paragraph indentation
\AMlayoutSetIndent%
%
% Enumerations
\setlist[enumerate,1]{label={\alph*)}}%
\setlist[itemize,1]{label={$\triangleright$}}%
%
% Package titlesec: chapter and section style
\titlespacing*\section{0pt}{15pt plus 4pt minus 2pt}{7pt plus 2pt minus 2pt}%
\titlespacing*\subsection{0pt}{10pt plus 2pt minus 2pt}{2pt plus 2pt minus 2pt}%
\titlespacing*\subsubsection{0pt}{10pt plus 2pt minus 2pt}{2pt plus 2pt minus 2pt}%
\titleformat{\section}{\sffamily\Large\bfseries}{\thesection}{1em}{}%
\titleformat{\subsection}{\sffamily\large\bfseries}{\thesubsection}{1em}{}%
\titleformat{\subsubsection}{\sffamily\normalsize\bfseries}{\thesubsubsection}{1em}{}%
%
% Package fancyhdr: header and footer style
\fancypagestyle{AMexercisep}{%
	\AtBeginDocument{%
		\renewcommand{\sectionmark}[1]{\markright{\thesection\ \ ##1}}%
	}%
    \fancyhf{}%
    \fancyhead[R]{\AMlogoTUM{0.7cm}{TUMBlue}}%
    \fancyhead[L]{\AMlogoAM{0.8cm}{AMLogoBlue1}{AMLogoBlue2}}%
    \fancyfoot[R]{\AMlangGerEng{Seite}{Page} \thepage}%
    \fancyfoot[L]{\subject}%
    \renewcommand{\headrulewidth}{0.5pt}%
    \renewcommand{\footrulewidth}{0.5pt}%
}%
%
% Floating environment names
\addto\captionsngerman{\renewcommand{\figurename}{Bild}}%
\crefname{figure}{Bild}{Bilder}%
%
% Set pagestyle for mainmatter
\pagestyle{AMexercisep}%
%
% References
\hypersetup{colorlinks=false}%
%
%
%
\else\fi% End-check if style AMexercise is enabled
%###################################################################################################
%--- END STYLE AMexercise --------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMdocument ------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMdocument% Start-check if style AMdocument is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMdocument.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%
%
%=============
%--- Style ---
%=============
% Paragraph indentation
\AMlayoutSetIndent%
%
% Caption style
\captionsetup{labelfont=bf,font=small}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMdocument.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if style AMdocument is enabled
%###################################################################################################
%--- END STYLE AMdocument --------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMbeamer --------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMbeamer% Start-check if style AMbeamer is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMbeamer.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%
\RequirePackage{transparent}%
%
%
%=============
%--- Style ---
%=============
% Document setup
% --------------
\geometry{left=1cm,right=1cm}%    set horizontal margins (ATTENTION: vertical margins remove footer!)
\ifdefined\optBeamerClassicFormat%
    \AMfontSetDocumentFontSizeXVIII% set font size to 18
\else%
    \AMfontSetDocumentFontSizeXIV% set font size to 14
\fi%
%
% Predefined fonts
% ----------------
\usefonttheme{professionalfonts}% fonts as defined in the document, beamer should not rewrite the font definitions
\captionsetup{labelfont=bf,font={sf,small}}%
\providecommand{\bibfont}{}%
\renewcommand*{\bibfont}{\footnotesize}%
%
% Predefined colors
% -----------------
\hypersetup{colorlinks=false}%
% Global definitions
\setbeamercolor{normal text}{fg=TUMBlack}%
\setbeamercolor{titlelike}{fg=TUMBlue}%
\setbeamercolor{separation line}{fg=TUMBlack}%
\setbeamercolor{fine separation line}{fg=TUMBlack}%
% Title page
\setbeamercolor{title}{fg=TUMBlue}%
\setbeamercolor{subtitle}{fg=TUMGray2}%
\setbeamercolor{author}{fg=TUMBlack}%
\setbeamercolor{institute}{fg=TUMBlack}%
\setbeamercolor{date}{fg=TUMBlack}%
% Part, section, subsection pages
\setbeamercolor{part page}{fg=TUMBlack}%
\setbeamercolor{section page}{fg=TUMBlack}%
\setbeamercolor{subsection page}{fg=TUMBlack}%
% Table of contents
\setbeamercolor{section in toc}{fg=TUMBlue}%
\setbeamercolor{subsection in toc}{fg=TUMBlack}%
\setbeamercolor{subsubsection in toc}{fg=TUMBlack}%
\setbeamercolor{section in toc shaded}{fg=TUMGray2}%
\setbeamercolor{subsection in toc shaded}{fg=TUMGray2}%
\setbeamercolor{subsubsection in toc shaded}{fg=TUMGray2}%
% Bibliography
\setbeamercolor{bibliography entry author}{fg=TUMBlack}%
\setbeamercolor{bibliography entry title}{fg=TUMBlack}%
\setbeamercolor{bibliography entry location}{fg=TUMBlack}%
\setbeamercolor{bibliography entry note}{fg=TUMBlack}%
\setbeamercolor{bibliography item}{fg=TUMBlack}%
% Note page
\setbeamercolor{note page}{fg=TUMBlack}%
% Headlines and footlines
\setbeamercolor{headline}{fg=TUMBlack}%
\setbeamercolor{footline}{fg=TUMBlack}%
\setbeamercolor{page number in head}{fg=TUMBlack}%
\setbeamercolor{page number in foot}{fg=TUMBlack}%
\setbeamercolor{author in head}{fg=TUMBlack}%
\setbeamercolor{author in foot}{fg=TUMBlack}%
\setbeamercolor{title in head}{fg=TUMBlack}%
\setbeamercolor{title in foot}{fg=TUMBlack}%
\setbeamercolor{date in head}{fg=TUMBlack}%
\setbeamercolor{date in foot}{fg=TUMBlack}%
\setbeamercolor{institute in head}{fg=TUMBlack}%
\setbeamercolor{institute in foot}{fg=TUMBlack}%
% Navigation symbols
\setbeamercolor{navigation symbols}{fg=TUMBlack}%
% Frame title
\setbeamercolor{frametitle}{fg=TUMBlue}%
\setbeamercolor{frametitle right}{fg=TUMBlue}%
\setbeamercolor{framesubtitle}{fg=TUMGray2}%
\setbeamercolor{frametitle continuation}{fg=TUMBlue}%
% Abstract
\setbeamercolor{abstract}{fg=TUMBlack}%
\setbeamercolor{abstract title}{fg=TUMBlue}%
% Lists
\setbeamercolor{itemize item}{fg=TUMBlack}%
\setbeamercolor{itemize subitem}{fg=TUMBlack}%
\setbeamercolor{itemize subsubitem}{fg=TUMBlack}%
\setbeamercolor{enumerate item}{fg=TUMBlack}%
\setbeamercolor{enumerate subitem}{fg=TUMBlack}%
\setbeamercolor{enumerate subsubitem}{fg=TUMBlack}%
\setbeamercolor{description item}{fg=TUMBlack}%
\setbeamercolor{item}{fg=TUMBlack}%
\setbeamercolor{item projected}{fg=TUMBlack}%
\setbeamercolor{subitem projected}{fg=TUMBlack}%
\setbeamercolor{subsubitem projected}{fg=TUMBlack}%
\setbeamercolor{subitem}{fg=TUMBlack}%
\setbeamercolor{subsubitem}{fg=TUMBlack}%
% Highlighting
\setbeamercolor{structure}{fg=TUMBlack}%
\setbeamercolor{local structure}{fg=TUMBlack}%
\setbeamercolor{alerted text}{fg=TUMOrange}%
% Blocks environments
\setbeamercolor{block title}{fg=TUMBlue}%
\setbeamercolor{block body}{fg=TUMBlack}%
\setbeamercolor{block title alerted}{fg=TUMOrange}%
\setbeamercolor{block body alerted}{fg=TUMBlack}%
\setbeamercolor{example text}{fg=TUMBlue}%
\setbeamercolor{block title example}{fg=TUMBlue}%
\setbeamercolor{block body example}{fg=TUMBlack}%
% Theorem environments
\setbeamercolor{qed symbol}{fg=TUMBlack}%
% Figures and tables
\setbeamercolor{caption}{fg=TUMBlack}%
\setbeamercolor{caption name}{fg=TUMBlack}%
% Verse, quotations and quotes
\setbeamercolor{verse}{fg=TUMBlack}%
\setbeamercolor{quotation}{fg=TUMBlack}%
\setbeamercolor{quote}{fg=TUMBlack}%
% Footnotes
\setbeamercolor{footnote}{fg=TUMBlack}%
\setbeamercolor{footnote mark}{fg=TUMBlack}%
% Mathematics
\setbeamercolor{math text}{fg=}% Use same as regular text color
\setbeamercolor{normal text in math text}{fg=}% Use same as regular text color
%
% Predefined items
% ----------------
\setbeamertemplate{itemize item}{\color{TUMBlack}$\vcenter{\hbox{$\bullet$}}$}%
\setbeamertemplate{itemize subitem}{\color{TUMBlack}$\vcenter{\hbox{$\circ$}}$}%
\setbeamertemplate{itemize subsubitem}{\color{TUMBlack}$\vcenter{\hbox{$\triangleright$}}$}%
\setbeamertemplate{enumerate item}{\color{TUMBlack}\insertenumlabel.}%
\setbeamertemplate{enumerate subitem}{\color{TUMBlack}\insertenumlabel.\insertsubenumlabel}%
\setbeamertemplate{enumerate subsubitem}{\color{TUMBlack}\insertenumlabel.\insertsubenumlabel.\insertsubsubenumlabel}%
\setbeamertemplate{bibliography item}[text]%
%
% Animations
% ----------
\setbeamercovered{transparent}%
%
% Frame title
% -----------
% Style
\setbeamertemplate{frametitle}{%
    % Print title
    {\huge\bfseries\insertframetitle}%
    % Print subtitle (if there is one)
    \ifx\insertframesubtitle\empty% check, if frame subtitle has been specified
        \\%
    \else%
        \\[0.6em]{\Large\color{TUMGray2}\insertframesubtitle}%
    \fi%
}%
%
% Header
% ------
\newcommand{\AMbeamerSetupHeader}[2][3cm]{\setbeamertemplate{headline}{\vspace{#1}\vspace{-\baselineskip}#2}}%
%
% Footer
% ------
\newcommand{\AMbeamerSetFooterText}[1]{\def\AMbeamerFooterText{#1}}%
\def\AMbeamerFooterText{}% initialize empty
\newcommand{\AMbeamerSetupFooterCD}{%
    \setbeamertemplate{footline}{\vspace*{5mm}{\hypersetup{allcolors=.}\scriptsize\hspace{1cm}\insertshortauthor\ | \insertshorttitle\ \ifx\AMbeamerFooterText\empty\else| \AMbeamerFooterText\fi\hfill \insertframenumber\hspace{1cm}}\vspace*{5mm}}%
    \setbeamertemplate{navigation symbols}{}%
}%
\newcommand{\AMbeamerSetupFooterSlideNumberOnly}{%
    \setbeamertemplate{footline}{\vspace*{5mm}{\hypersetup{allcolors=.}\scriptsize\hfill \insertframenumber\hspace{1cm}}\vspace*{5mm}}%
    \setbeamertemplate{navigation symbols}{}%
}%
%
% Notes
% -----
\setbeamertemplate{note page}{%
    \vspace*{2cm}%
    \begin{tabular}{ll}%
        \large\bfseries \color{TUMBlue}Notes of Slide \insertframenumber: & \large\insertframetitle\\%
         & \color{TUMGray2}\insertframesubtitle%
    \end{tabular}\\[1cm]%
    \parbox{0.4\paperwidth}{\insertnote}%
    \AMlayoutPutAtNorthEast{15mm}{0.3\paperheight}{\tikz[overlay]\node[fill=TUMBlack,inner sep=1mm, anchor=text, rectangle] {\insertslideintonotes{0.4}};\phantom{\insertslideintonotes{0.4}}}%
}%
%
% Parts
% -----
% Store part title to variables
\newcommand*\parttitle{}%
\newcommand*\partshorttitle{}%
\let\origpart\part%
\renewcommand*{\part}[2][]{%
   \ifx\\#1\\%
      \renewcommand*\parttitle{#2}%
      \renewcommand*\partshorttitle{#2}%
      \origpart{#2}%
   \else%
      \renewcommand*\parttitle{#2}%
      \renewcommand*\partshorttitle{#1}%
      \origpart[#1]{#2}%
   \fi%
}%
%
% Setup part page
\AtBeginPart{%
    \begin{frame}%
        \vfill%%
        \begin{center}%
            \begingroup%
                \Large\AMlangGerEng{Teil}{Part} \Roman{part}\\[2em]%
                \huge\color{TUMBlue}\bfseries\parttitle%
            \endgroup%
        \end{center}%
        \vfill%
    \end{frame}%
}%
%
% Sections
% --------
% Section page
\AtBeginSection[]{%
    \begin{frame}{\AMlangGerEng{Übersicht}{Outline}}%
        \tableofcontents[currentsection]%
    \end{frame}%
}%
%
%
%=================
%--- Titlepage ---
%=================
% Default titlepage
% -----------------
\defbeamertemplate*{title page}{customized}[1][]{%
    \begin{center}%
        \vspace{1cm}%
        % Print title
        \ifx\inserttitle\empty\else{%
            \huge\color{TUMBlue}\bfseries\inserttitle\\[0.5em]%
        }\fi%
        % Print subtitle
        \ifx\insertsubtitle\empty\else{%
            \large\color{TUMGray2}\insertsubtitle\\%
        }\fi%
        % Print author
        \ifx\insertauthor\empty\else{%
            \vspace{1cm}\insertauthor\\%
        }\fi%
        % Print institute
        \ifx\insertinstitute\empty\else{%
            \itshape\insertinstitute\\%
        }\fi%
        % Print date
        \ifx\insertdate\empty\else{%
            \vspace{0.5cm}\insertdate\\%
        }\fi%
    \end{center}%
}%
\newcommand{\AMbeamerTitlePageDefault}{%
    \begingroup%
    \setbeamertemplate{footline}{}% skip footline on titlepage
    \begin{frame}{}%
        \maketitle%
    \end{frame}%
    \endgroup%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMbeamer.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%=======================
%--- Option handling ---
%=======================
% Check, if either lecture of exercise slide style is chosen
% ----------------------------------------------------------
\ifdefined\optLectureSlides
    \def\optTeachingSlides{x}%
\else\fi%
\ifdefined\optExerciseSlides
    \def\optTeachingSlides{x}%
\else\fi%
%
%
%================
%--- Packages ---
%================
\ifdefined\optTeachingSlides%
    \RequirePackage{multirow}%
    \RequirePackage{fancybox}%
    \RequirePackage{listings}%     Typeset code
    \RequirePackage{empheq}%
    \RequirePackage{tcolorbox}%    Colored and framed text boxes
    \tcbuselibrary{skins}%         Load skin library
\else\fi%
%
%
%=============
%--- Style ---
%=============
\ifdefined\optTeachingSlides%
    % Document setup
    \geometry{left=2cm,right=2cm}% set horizontal margins (ATTENTION: vertical margins remove footer!)
    %
    % Frame title
    \setbeamercolor{frametitle}{fg=TUMBlack}%
    \setbeamertemplate{frametitle}{%
        % Print title
        \hspace{-1cm}{\Large\bfseries\insertframetitle}%
        % Print subtitle (if there is one)
        \ifx\insertframesubtitle\empty% check, if frame subtitle has been specified
            \\%
        \else%
            \\[0.6em]\hspace{-1cm}{\large\color{TUMGray2}\insertframesubtitle}%
        \fi%
    }%
    %
    % Redefine \emph
    \renewcommand<>{\emph}[1]{{\color{TUMOrange}{\only#2{\itshape}#1}}}%
    %
    % Emphasize equations
    \newtcbox{\empheqinnerbox}[1][]{reset,nobeforeafter,tcbox raise base,boxrule=1pt,colframe=TUMBlue,colback=white,arc=0pt,coltext=TUMBlue,#1}%
    \empheqset{innerbox=\empheqinnerbox}%
    %
    % Theorems
    \newtcolorbox{AMtheorem}[1][]{fontupper=\itshape,coltext=TUMBlue,boxrule=1pt,colframe=TUMBlue,#1}%
\else\fi%
%
%
%=================
%--- Titlepage ---
%=================
% Student theses titlepage
% ------------------------
\newcommand{\AMbeamerTitlePageStudentThesis}{%
    \begingroup%
    \setbeamertemplate{footline}{}% skip footline on titlepage
    \begin{frame}{}%
        \begin{center}%
            \vspace{1cm}%
            {\Huge\color{TUMBlue}\bfseries\inserttitle\\[0.2em]}%
            {\Large\color{TUMGray2}\PresentationType, \PresentationThesisType\\}%
            {\vspace{1cm}\insertauthor\\}%
            {\vspace{0.5cm}\insertdate\\}%
        \end{center}%
        \AMlayoutPutAtSouthWest{1cm}{1cm}{%
            \small%
            \begin{tabular}{@{}ll}%
                \AMlangSupervisor: & \itshape\PresentationSupervisor\\%
                \AMlangExaminer: & \itshape\PresentationExaminer%
            \end{tabular}%
        }%
    \end{frame}%
    \endgroup%
}%
%
% Lecture slides titlepage
% ------------------------
\newcommand{\AMbeamerTitlePageLectureSlides}[2]{%
    \begingroup%
    \setbeamertemplate{footline}{}% skip footline on titlepage
    \begin{frame}{}%
        \begin{center}%
            \ifx&#2&%
                \vspace{1cm}%
            \else%
                \vspace{-1cm}%
            \fi%
            {\Huge\color{TUMBlue}\bfseries\inserttitle\\[0.2em]}%
            {\Large\color{TUMBlack} \insertsubtitle\\[0.3em]}%
            {\color{TUMGray2}\AMlangGerEng{Folien zur Vorlesung}{Lecture Slides}, #1\\}%
            \ifx&#2&\else%
                \vspace{5mm}%
                {#2\\}%
                \vspace{5mm}%
            \fi%
            {\vspace{1cm}\insertauthor\\}%
            {\vspace{0.3cm}\insertdate\\}%
        \end{center}%
    \end{frame}%
    \endgroup%
}%
%
% Exercise slides titlepage
% -------------------------
\newcommand{\AMbeamerTitlePageExerciseSlides}[2]{%
    \begingroup%
    \setbeamertemplate{footline}{}% skip footline on titlepage
    \begin{frame}{}%
        \begin{center}%
            \ifx&#2&%
                \vspace{1cm}%
            \else%
                \vspace{-1cm}%
            \fi%
            {\Huge\color{TUMBlue}\bfseries\inserttitle\\[0.2em]}%
            {\Large\color{TUMBlack} \insertsubtitle\\[0.3em]}%
            {\color{TUMGray2}\AMlangGerEng{Folien zur Zentralübung}{Exercise Class Slides}, #1\\}%
            \ifx&#2&\else%
                \vspace{5mm}%
                {#2\\}%
                \vspace{5mm}%
            \fi%
            {\vspace{1cm}\insertauthor\\}%
            {\vspace{0.3cm}\insertdate\\}%
        \end{center}%
    \end{frame}%
    \endgroup%
}%
%
%
%
\else\fi% End-check if style AMbeamer is enabled
%###################################################################################################
%--- END STYLE AMbeamer ----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMposter --------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMposter% Start-check if style AMposter is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMposter.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%=============
%--- Style ---
%=============
% Set page layout
\AMlayoutSetAzero%
\geometry{margin=0pt}%
\AMfontSetDocumentFontSizeXLIV%
%
% Bibliography style
\ifdefined\BackendSpecified%
    \AtBeginBibliography{\footnotesize}%
\else\fi%
%
%
%================
%--- COMMANDS ---
%================
\newcommand{\AMposterDrawGrid}{%
    \AMlayoutPutAtNorthWest{0mm}{0mm}{\AMtikzGrid[TUMBlack!5!TUMWhite,line width=1mm]{10mm}{10mm}{85}{119}}%
    \AMlayoutPutAtNorthWest{0mm}{0mm}{\AMtikzGrid[TUMBlack!20!TUMWhite,line width=1mm]{50mm}{50mm}{17}{24}}%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMposter.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- COMMANDS ---
%================
% Grid and margins
% ----------------
\newcommand{\AMposterDrawBorders}{%
    \AMlayoutPutAtSouthWest{0mm}{0mm}{%
        \AMtikzExternalizeSkipNext%
        \begin{tikzpicture}[inner sep=0pt, outer sep=0pt]%
            \draw[red,line width=1mm]%
            (0mm,40mm) -- (841mm,40mm)%
            (0mm,940mm) -- (841mm,940mm)%
            (0mm,1108mm) -- (841mm,1108mm)%
            (0mm,1148mm) -- (841mm,1148mm)%
            (40mm,0mm) -- (40mm,1189mm)%
            (210mm,0mm) -- (210mm,1189mm)%
            (230mm,0mm) -- (230mm,1189mm)%
            (800mm,0mm) -- (800mm,1189mm);%
        \end{tikzpicture}%
    }%
}%
%
% Title and Header
% ----------------
\newcommand{\AMposterTitle}[2]{%
    \AMlayoutPutAtSouthWest{40mm}{1015mm}{\parbox{750mm}{\raggedright\Huge\color{TUMBlue} #1}}%
    \AMlayoutPutAtNorthWest{40mm}{185mm}{\parbox{750mm}{\raggedright\Large\color{TUMGray2} #2}}%
}%
\newcommand{\AMposterTitleWithSnake}[2]{%
    \AMlayoutPutAtSouthWest{40mm}{1015mm}{\parbox{525mm}{\raggedright\Huge\color{TUMBlue} #1}}%
    \AMlayoutPutAtNorthWest{40mm}{192mm}{\parbox{750mm}{\raggedright\Large\color{TUMGray2} #2}}%
    \AMlayoutPutAtNorthWest{40mm}{118mm}{\AMtikzAMsnake{761mm}{60mm}{541mm}{1mm}{TUMBlue}}%
    \AMlayoutPutAtNorthWest{40mm}{124mm}{\AMtikzAMsnake{761mm}{60mm}{544mm}{1mm}{TUMBlue}}%
}%
%
% Content
% -------
\newcommand{\AMposterAbstract}[1]{%
    \AMlayoutPutAtSouthWest{40mm}{40mm}{\tikz \fill [black!10!white] (0mm,0mm) rectangle (170mm,900mm);}%
    \AMlayoutPutAtNorthWest{50mm}{258mm}{\fontsize{35}{38}\selectfont \parbox{150mm}{\hbadness=10000 #1}}%
}%
\newcommand{\AMposterContact}[3]{%
    \AMlayoutPutAtSouthWest{50mm}{80mm}{\includegraphics[width=100mm]{#1}}%
    \AMlayoutPutAtSouthWest{50mm}{62mm}{\footnotesize #2}%
    \AMlayoutPutAtSouthWest{50mm}{47mm}{\footnotesize #3}%
}%
\newcommand{\AMposterBox}[6]{%
    \AMlayoutPutAtNorthWest{\dimexpr 230mm + #1}{\dimexpr 249mm + #2}{\tikz \fill [draw=TUMBlack!10!TUMWhite,rounded corners=2mm,line width=2mm,fill=none] (0mm,0mm) rectangle (#3,#4);}%
    \AMlayoutPutAtNorthWest{\dimexpr 240mm + #1}{\dimexpr 259mm + #2}{\parbox{\dimexpr #3 - 20mm}{\raggedright\hbadness=10000 {\LARGE \color{TUMBlue}#5}\\[0.5em]{\large #6}}}%
}%
%
%
%
\else\fi% End-check if style AMposter is enabled
%###################################################################################################
%--- END STYLE AMposter ----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMphdprepublication ---------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMphdprepublication% Start-check if style AMphdprepublication is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMphdprepublication.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%
\RequirePackage{fancyhdr}%
%
%=============
%--- Style ---
%=============
% Set default layout
\geometry{left=3cm,right=3cm,top=2.5cm,bottom=3cm,headheight=14pt}%
%
% Paragraph indentation
\AMlayoutSetIndent%
%
% Package titlesec: section style
\titlespacing*\section{0pt}{30pt plus 4pt minus 2pt}{16pt plus 2pt minus 2pt}%
\titlespacing*\subsection{0pt}{20pt plus 4pt minus 2pt}{12pt plus 2pt minus 2pt}%
\titlespacing*\subsubsection{0pt}{10pt plus 4pt minus 2pt}{2pt plus 2pt minus 2pt}%
\titleformat{\section}{\color{TUMBlack}\sffamily\Large\bfseries}{\thesection}{1em}{}%
\titleformat{\subsection}{\color{TUMBlack}\sffamily\large\bfseries}{\thesubsection}{1em}{}%
\titleformat{\subsubsection}{\color{TUMBlack}\sffamily\normalsize\bfseries}{\thesubsubsection}{1em}{}%
%
% Package fancyhdr: header and footer style
\def\AMfooterAuthor{Author}%
\def\AMfooterDate{Date}%
\fancypagestyle{AMphdprepublicationp}{%
	\AtBeginDocument{%
		\renewcommand{\sectionmark}[1]{\markright{\thesection\ \ ##1}}%
	}%
	\fancyhf{}%
	\fancyhead[L,R]{}%
	\fancyfoot[L]{\textit{\sffamily\footnotesize\AMfooterAuthor, \AMlangChair, \AMfooterDate}}%
	\fancyfoot[R]{\textit{\sffamily\thepage}}%
	\renewcommand{\headrulewidth}{0pt}%
	\renewcommand{\footrulewidth}{0pt}%
}%
%
% Set pagestyle
\pagestyle{AMphdprepublicationp}%
%
% Change plain style
\fancypagestyle{plain}{%
	\fancyhf{}% clear all header and footer fields
	\fancyfoot[C]{\normalcolor\large\bfseries\sffamily\thepage}%
	\renewcommand{\headrulewidth}{0pt}%
	\renewcommand{\footrulewidth}{0pt}%
}%
%
% Caption style
\captionsetup{labelfont=bf,font={sf,small}}%
%
% Bibliography style
\ExecuteBibliographyOptions{sorting=none}%
%
%
%=================
%--- Titlepage ---
%=================
\newcommand{\AMphdprepublicationTitlePage}[2]{%
	\def\AMfooterAuthor{#1}%
	\def\AMfooterDate{#2}%
    \AMrefSetPDFMetadata{\AMlangGerEng{Vorveröffentlichungsliste}{Pre-Publication List}}{}{#1}{}{}{}%
    \AMlayoutHeaderCDSchoolTUMLogoOnly%
	\par%
	\noindent\Huge\textbf{\color{TUMBlue}\AMlangGerEng{Vorveröffentlichungsliste}{Pre-Publication List}}%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMphdprepublication.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if style AMphdprepublication is enabled
%###################################################################################################
%--- END STYLE AMphdprepublication -----------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMphdthesis -----------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMphdthesis% Start-check if style AMphdthesis is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMphdthesis.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%=================
%--- Titlepage ---
%=================
% Titlepage (TUM corporate design)
% ---------
\newcommand{\AMphdthesisTitlePage}[9]{{%
	\thispagestyle{empty}%
	\newgeometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}%
	\AMlayoutPutAtNorthEast{1.5cm}{1.5cm}{\AMlogoTUM{1cm}{TUMBlue}}%
	\sffamily%
	\vskip1.4cm%
	\noindent\Large \AMlangForceGerman{\AMlangTUM}\\\AMlangForceGerman{\AMlangSchool}\par%
    % Title
	\vskip2cm%
	\noindent\huge\textbf{#1}\par%
    % Subtitle
	\noindent\textit{#2}%
	\vskip1cm%
	% Author
	\noindent{\color{TUMBlack}\bfseries\fontsize{13pt}{18pt}\selectfont #3\par}%
	\vskip1cm%
	%
	\normalsize%
	\noindent%
	Vollständiger Abdruck der von der \AMlangForceGerman{\AMlangSchoolED} der Technischen Universität München zur Erlangung eines\par\vskip1em%
	\noindent\textbf{#4}\par\vskip1em%
	\noindent genehmigten Dissertation.\par%
	\vfill%
	\noindent\textbf{Vorsitz:} #5\par%
	\vskip1em%
	\noindent\textbf{Prüfer*innen der Dissertation:}\par\vskip0.3em 1. #6\par\vskip0.3em 2. #7\par%
	\vskip1.5cm%
	\noindent Die Dissertation wurde am \AMlangForceGerman{#8} bei der Technischen Universität München eingereicht und durch die \AMlangForceGerman{\AMlangSchoolED} am \AMlangForceGerman{#9} angenommen.%
	\vskip2cm%
	\cleardoublepage%
	\restoregeometry%
}}%
%
%
\newcommand{\AMphdthesisTitlePageFemale}[9]{{%
		\thispagestyle{empty}%
		\newgeometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}%
		%\AMlayoutPutAtNorthWest{2.5cm}{1.5cm}{\AMlogoDepartment{1cm}{TUMBlue}}%
		\AMlayoutPutAtNorthEast{1.5cm}{1.5cm}{\AMlogoTUM{1cm}{TUMBlue}}%
		\sffamily%
		\vskip1.4cm%
		\noindent\Large \AMlangForceGerman{\AMlangTUM}\\\AMlangForceGerman{\AMlangSchool}\par%
		% Title
		\vskip2cm%
		\noindent\huge\textbf{#1}\par%
		% Subtitle
		\noindent\textit{#2}%
		\vskip1cm%
		% Author
		\noindent{\color{TUMBlack}\bfseries\fontsize{13pt}{18pt}\selectfont #3\par}%
		\vskip1cm%
		%
		\normalsize%
		\noindent%
		Vollständiger Abdruck der von der \AMlangForceGerman{\AMlangSchoolED} der Technischen Universität München zur Erlangung des akademischen Grades einer\par\vskip1em%
		\noindent\textbf{#4}\par\vskip1em%
		\noindent genehmigten Dissertation.\par%
		\vfill%
		\noindent\textbf{Vorsitz:} #5\par%
		\vskip1em%
		\noindent\textbf{Prüfer*innen der Dissertation:}\par\vskip0.3em 1. #6\par\vskip0.3em 2. #7\par%
		\vskip1.5cm%
		\noindent Die Dissertation wurde am \AMlangForceGerman{#8} bei der Technischen Universität München eingereicht und durch die \AMlangForceGerman{\AMlangSchoolED} am \AMlangForceGerman{#9} angenommen.%
		\vskip2cm%
		\cleardoublepage%
		\restoregeometry%
}}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMphdthesis.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%
\RequirePackage{fancyhdr}%
%
%=============
%--- Style ---
%=============
% Set default layout
\geometry{left=3cm,right=2.5cm,top=2.5cm,bottom=3cm,headheight=14pt}%
%
% Paragraph indentation
\AMlayoutSetIndent%
%
% Package titlesec: chapter and section style
\titlespacing*\chapter{0pt}{50pt plus 4pt minus 2pt}{40pt plus 2pt minus 2pt}%
\titlespacing*\section{0pt}{30pt plus 4pt minus 2pt}{16pt plus 2pt minus 2pt}%
\titlespacing*\subsection{0pt}{26pt plus 4pt minus 2pt}{14pt plus 2pt minus 2pt}%
\titlespacing*\subsubsection{0pt}{20pt plus 4pt minus 2pt}{6pt plus 2pt minus 2pt}%
\titleformat{\chapter}[display]{\color{TUMBlack}\sffamily\huge\bfseries}{\large\chaptertitlename\ \thechapter}{0.3em}{}%
\titleformat{\section}{\color{TUMBlack}\sffamily\Large\bfseries}{\thesection}{1em}{}%
\titleformat{\subsection}{\color{TUMBlack}\sffamily\large\bfseries}{\thesubsection}{1em}{}%
\titleformat{\subsubsection}{\color{TUMBlack}\sffamily\normalsize\bfseries}{\thesubsubsection}{1em}{}%
%
% Package fancyhdr: header and footer style
\fancypagestyle{AMphdthesisp}{%
	\AtBeginDocument{%
		\renewcommand{\sectionmark}[1]{\markright{\thesection\ \ ##1}}%
		\renewcommand{\chaptermark}[1]{\markboth{\thechapter\ \ ##1}{}}%
	}%
	\fancyhf{}%
	\fancyhead[LE,RO]{\normalcolor\large\bfseries\sffamily\thepage}%
	\fancyhead[LO]{\normalcolor\normalsize\sffamily\nouppercase{\rightmark}}%
	\fancyhead[RE]{\normalcolor\normalsize\sffamily\nouppercase{\leftmark}}%
	\renewcommand{\headrulewidth}{0.4pt}%
	\renewcommand{\footrulewidth}{0pt}%
}%
%
% Set pagestyle for mainmatter
\pagestyle{AMphdthesisp}%
%
% Change plain style
\fancypagestyle{plain}{%
	\fancyhf{} % clear all header and footer fields
	\fancyfoot[C]{\normalcolor\large\bfseries\sffamily\thepage}%
	\renewcommand{\headrulewidth}{0pt}%
	\renewcommand{\footrulewidth}{0pt}}%
%
% Hyperref style (no color)
\hypersetup{linkcolor=black,urlcolor=black}%
%
% Caption style
\captionsetup{labelfont=bf,font={sf,small}}%
%
% Contents/Figures/Tables
\newcommand{\AMphdthesisPrintTableOfContents}{%
	\AMlayoutNoIndent%
	\tableofcontents%
	\AMlayoutSetIndent%
	\cleardoublepage%
}%
%
\newcommand{\AMphdthesisTitlePageCustomHabilitation}[8]{{%
		\thispagestyle{empty}%
		\newgeometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}%
		%\AMlayoutPutAtNorthWest{2.5cm}{1.5cm}{\AMlogoDepartment{1cm}{TUMBlue}}%
		\AMlayoutPutAtNorthEast{1.5cm}{1.5cm}{\AMlogoTUM{1cm}{TUMBlue}}%
		\sffamily%
		\vskip1.4cm%
		\noindent\Large \AMlangForceGerman{\AMlangTUM}\\\AMlangForceGerman{\AMlangSchool}\\\AMlangForceGerman{\AMlangChair}\par%
		% Title
		\vskip2cm%
		\noindent\huge\textbf{#1}\par%
		% Subtitle
		\noindent\textit{#2}%
		\vskip1cm%
		% Author
		\noindent{\color{TUMBlack}\bfseries\fontsize{13pt}{18pt}\selectfont #3\par}%
		\vskip1cm%
		%
		\normalsize%
		\noindent%
		Vollständiger Abdruck der zur Erlangung der Lehrbefähigung an der \AMlangForceGerman{\AMlangSchoolED} der Technischen Universität 
		München für das\par\vskip1em%
		\noindent\textbf{Fachgebiet #4}\par\vskip1em%
		\noindent eingereichten Habilitationsschrift.\par%
		\vfill%
		\noindent\textbf{Mentorat:}\par\vskip0.3em 1. #5\par\vskip0.3em 2. #6\par\vskip0.3em 3. #7\par%
		\vskip1.5cm%
		\noindent Die Habiliation wurde am \AMlangForceGerman{#8} bei der 
		Technischen Universit\"at M\"unchen eingereicht.%
		\vskip2cm%
		\cleardoublepage%
		\restoregeometry%
}}%
%
%
%
\else\fi% End-check if style AMphdthesis is enabled
%###################################################################################################
%--- END STYLE AMphdthesis -------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMempty ---------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMempty% Start-check if style AMempty is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMempty.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMempty.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if style AMempty is enabled
%###################################################################################################
%--- END STYLE AMempty -----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMstudentthesis -------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMstudentthesis% Start-check if style AMstudentthesis is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMstudentthesis.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{ifthen}%
%
%
%=================
%--- Titlepage ---
%=================
% Titlepage (TUM corporate design)
% ---------
\newcommand{\AMstudentthesisTitlePageCD}[8]{%
	\AMstudentthesisTitlePageFrontPage{#1}{#2}{#3}{#4}{#5}{#6}{#7}{#8}%
	\cleardoublepage%
	\thispagestyle{empty}%
	% Disclaimer at end of document with thesis date
	\AtEndDocument{\AMstudentthesisDeclaration{#3}{#6}}%
}%
%
\newcommand{\AMstudentthesisTitlePageFrontPage}[8]{%
	\thispagestyle{empty}%
	% Logo
	\AMlayoutPutAtNorthEast{2cm}{1.5cm}{\AMlogoTUM{1cm}{TUMBlue}}%
	\vskip3.5cm\noindent%
	% Title in primary language
	\noindent{\color{TUMBlue}\sffamily\bfseries\fontsize{24pt}{24pt}\selectfont\AMlangGerEng{#1}{#2}\par}%
	\vskip1em\noindent%
	% Title in secondary language
	\noindent{\color{TUMGray3}\sffamily\bfseries\fontsize{20pt}{20pt}\selectfont\AMlangGerEng{#2}{#1}\par}%
	\vskip2cm\noindent%
	\vfill%
	% Subject
	{\sffamily%
		\noindent\fontsize{14pt}{18pt}\selectfont #7\par%
		\noindent\fontsize{16pt}{18pt}\selectfont #8\par%
		\noindent\fontsize{12pt}{16pt}\selectfont \AMlangGerEng{an der \AMlangSchool\ der Technischen Universität München}{at the \AMlangSchool\ of the Technical University of Munich}\par%
	}\par%
	\vskip2cm\noindent%
	% Info
	\AMstudentthesisTitlePageFrontPageInfoTable{#4}{#5}{#3}{#6}%
	\vskip1cm%
}%
%
\newcommand{\AMstudentthesisTitlePageFrontPageInfoTable}[4]{%
	{\sffamily\fontsize{12pt}{14pt}\selectfont\noindent%
	\begin{tabular}{@{}p{0.22\textwidth}p{0.75\textwidth}}%
		\bfseries \AMlangGerEng{Themenstellende/r}{Supervisor} & #1 \newline\AMlangChair\\\\%
		\ifthenelse{\equal{#2}{}}{}{\bfseries \AMlangGerEng{Betreuer/Betreuerin}{Advisor} & #2 \newline\AMlangChair\\\\}% (Optional)
		\bfseries \AMlangGerEng{Eingereicht von}{Submitted by} & #3\\\\%
		\bfseries \AMlangGerEng{Eingereicht am}{Submitted on} & #4 in \AMaddressCityChair\\%
	\end{tabular}}%
}%
%
\newcommand{\AMstudentthesisDeclaration}[2]{%
	\chapter*{\AMlangGerEng{Erklärung}{Disclaimer}}%
	\AMlangGerEng{Ich versichere hiermit, dass ich die von mir eingereichte Arbeit selbstständig verfasst und keine anderen als die angegebenen Quellen und Hilfsmittel benutzt habe.}{I hereby declare that this thesis is entirely the result of my own work except where otherwise indicated. I have only used the resources given in the list of references.}%
	\vskip4\baselineskip%
	\noindent \AMaddressCityChair, #2\hfill $\overline{\makebox[60mm]{\centering\footnotesize(\AMlangGerEng{Unterschrift}{Signature})}}$%
}%
%
\newcommand{\AMstudentthesisTitlePageCDDiplomarbeit}[6]{%
    \AMstudentthesisTitlePageCD{#1}{#2}{#3}{#4}{#5}{#6}{\AMlangScientificWorkForObtainingAcademicDegree}{Diplom Ingenieur (Dipl.-Ing.)}%
}%
\newcommand{\AMstudentthesisTitlePageCDBachelorsThesis}[6]{%
    \AMstudentthesisTitlePageCD{#1}{#2}{#3}{#4}{#5}{#6}{\AMlangScientificWorkForObtainingAcademicDegree}{Bachelor of Science (B.Sc.)}%
}
\newcommand{\AMstudentthesisTitlePageCDSemesterThesis}[6]{%
    \AMstudentthesisTitlePageCD{#1}{#2}{#3}{#4}{#5}{#6}{}{\AMlangSemesterThesis}%
}%
\newcommand{\AMstudentthesisTitlePageCDMasterThesis}[6]{%
    \AMstudentthesisTitlePageCD{#1}{#2}{#3}{#4}{#5}{#6}{\AMlangScientificWorkForObtainingAcademicDegree}{Master of Science (M.Sc.)}%
}%
\newcommand{\AMstudentthesisTitlePageCDIDP}[6]{%
    \AMstudentthesisTitlePageCD{#1}{#2}{#3}{#4}{#5}{#6}{}{\AMlangInterdisciplinaryProject}%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMstudentthesis.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%
\RequirePackage{fancyhdr}%
%
%=============
%--- Style ---
%=============
% Set default layout
\geometry{left=2.5cm,right=3cm,top=2.5cm,bottom=3cm,headsep=1cm,headheight=14pt}%
%
% Paragraph indentation
\AMlayoutSetIndent%
%
% Package titlesec: chapter and section style
\titlespacing*\chapter{0pt}{50pt plus 4pt minus 2pt}{40pt plus 2pt minus 2pt}%
\titlespacing*\section{0pt}{30pt plus 4pt minus 2pt}{16pt plus 2pt minus 2pt}%
\titlespacing*\subsection{0pt}{26pt plus 4pt minus 2pt}{14pt plus 2pt minus 2pt}%
\titlespacing*\subsubsection{0pt}{20pt plus 4pt minus 2pt}{6pt plus 2pt minus 2pt}%
\titleformat{\chapter}[display]{\color{TUMBlue}\sffamily\huge\bfseries}{\chaptertitlename\ \thechapter}{1em}{}%
\titleformat{\section}{\color{TUMBlue}\sffamily\Large\bfseries}{\thesection}{1em}{}%
\titleformat{\subsection}{\color{TUMBlue}\sffamily\large\bfseries}{\thesubsection}{1em}{}%
\titleformat{\subsubsection}{\color{TUMBlue}\sffamily\normalsize\bfseries}{\thesubsubsection}{1em}{}%
%
%% Package fancyhdr: header and footer style
\fancypagestyle{AMstudentthesisp}{%
\AtBeginDocument{%
	\renewcommand{\sectionmark}[1]{\markright{\thesection\ \ ##1}}%
	\renewcommand{\chaptermark}[1]{\markboth{\thechapter\ \ ##1}{}}%
}%
\fancyhf{}%
\fancyhead[LE,RO]{\normalcolor\normalsize\bfseries\sffamily\thepage}%
\fancyhead[LO]{\normalcolor\normalsize\itshape\sffamily\nouppercase{\rightmark}}%
\fancyhead[RE]{\normalcolor\normalsize\itshape\sffamily\nouppercase{\leftmark}}%
\renewcommand{\headrulewidth}{0pt}%
\renewcommand{\footrulewidth}{0pt}%
}%
%
% Set pagestyle for mainmatter
\pagestyle{AMstudentthesisp}%
%
% First page of chapter empty instead of plain style
\patchcmd{\chapter}{plain}{empty}{}{}%
%
% Hyperref style (no color)
\hypersetup{linkcolor=black,urlcolor=black}%
%
% Caption style
\captionsetup{labelfont=bf,font={sf,small}}%
%
%=================
%--- Titlepage ---
%=================
% Titlepage (AM custom)
% ---------
\newcommand{\AMTitlePageStudentThesisCustomDiplomarbeit}[6]{%
    \AMlayoutPutAtNorthWest{2.5cm}{1.5cm}{\AMlogoTercetChair{1cm}}%
    \AMstudentthesisTitlePageCDDiplomarbeit{#1}{#2}{#3}{#4}{#5}{#6}%
}%
\newcommand{\AMstudentthesisTitlePageCustomBachelorsThesis}[6]{%
    \AMlayoutPutAtNorthWest{2.5cm}{1.5cm}{\AMlogoTercetChair{1cm}}%
    \AMstudentthesisTitlePageCDBachelorsThesis{#1}{#2}{#3}{#4}{#5}{#6}%
}%
\newcommand{\AMstudentthesisTitlePageCustomSemesterThesis}[6]{%
    \AMlayoutPutAtNorthWest{2.5cm}{1.5cm}{\AMlogoTercetChair{1cm}}%
    \AMstudentthesisTitlePageCDSemesterThesis{#1}{#2}{#3}{#4}{#5}{#6}%
}%
\newcommand{\AMstudentthesisTitlePageCustomMastersThesis}[6]{%
    \AMlayoutPutAtNorthWest{2.5cm}{1.5cm}{\AMlogoTercetChair{1cm}}%
    \AMstudentthesisTitlePageCDMasterThesis{#1}{#2}{#3}{#4}{#5}{#6}%
}%
\newcommand{\AMstudentthesisTitlePageCustomIDP}[6]{%
    \AMlayoutPutAtNorthWest{2.5cm}{1.5cm}{\AMlogoTercetChair{1cm}}%
    \AMstudentthesisTitlePageCDIDP{#1}{#2}{#3}{#4}{#5}{#6}%
}%
\newcommand{\AMstudentthesisTitlePageCustomProjectSeminarReport}[6]{%
	\AMlayoutPutAtNorthWest{2.5cm}{1.5cm}{\AMlogoTercetSchool{1cm}}%
	\AMstudentthesisTitlePageFrontPageProjectSeminarReport{#1}{#2}{#3}{#4}{#5}{#6}%
	\cleardoublepage%
	\thispagestyle{empty}%
}%
%
%
\newcommand{\AMstudentthesisTitlePageFrontPageProjectSeminarReport}[6]{%
	\thispagestyle{empty}%
	% Logo
	\AMlayoutPutAtNorthEast{2cm}{1.5cm}{\AMlogoTUM{1cm}{TUMBlue}}%
	\vskip3.5cm\noindent%
	% Title of the project seminar
	\noindent{\color{TUMBlue}\sffamily\bfseries\fontsize{24pt}{24pt}\selectfont #1\par}%
	\vskip1em\noindent%
	% Title of the specific topic
	\noindent{\color{TUMGray3}\sffamily\bfseries\fontsize{20pt}{20pt}\selectfont #2\par}%
	\vskip2cm\noindent%
	\vfill%
	% Subject
	{\sffamily%
		\noindent\fontsize{14pt}{18pt}\selectfont \AMlangGerEng{Abschlussbericht über das Projektseminar}{Final Report about the Project Seminar}\par%
		\noindent\fontsize{12pt}{16pt}\selectfont \AMlangGerEng{an der \AMlangSchool\ der Technischen Universität München}{at the \AMlangSchool\ of the Technical University of Munich}\par%
	}\par%
	\vskip2cm\noindent%
	% Info
	\AMstudentthesisTitlePageFrontPageInfoTableProjectSeminarReport{#3}{#4}{#5}{#6}%
	\vskip1cm%
}%
%
\newcommand{\AMstudentthesisTitlePageFrontPageInfoTableProjectSeminarReport}[4]{%
	{\sffamily\fontsize{12pt}{14pt}\selectfont\noindent%
		\begin{tabular}{@{}p{0.22\textwidth}p{0.75\textwidth}}%
			\bfseries \AMlangGerEng{Eingereicht von}{Submitted by} & #1\\\\%
			\bfseries \AMlangGerEng{Betreut von}{Supervised by} & #2\\\\%
			\bfseries \AMlangGerEng{Zeitraum}{Period} & #3~\AMlangGerEng{bis}{until}~#4 \\%
	\end{tabular}}%
}%
%
%===================
%--- Frontmatter ---
%===================
\newcommand{\AMstudentthesisAbstract}[2]{%
	\thispagestyle{empty}%
	\vspace*{2cm}%
	\section*{\color{TUMBlack}Abstract}#1%
	\vspace*{1cm}%
	\section*{\color{TUMBlack}Zusammenfassung}#2%
	\vfill%
	\cleardoublepage%
}%
\newcommand{\AMstudentthesisPrintTableOfContents}{%
	\AMlayoutNoIndent%
	\tableofcontents%
	\AMlayoutSetIndent%
	\cleardoublepage%
}%
%
%
%
\else\fi% End-check if style AMstudentthesis is enabled
%###################################################################################################
%--- END STYLE AMstudentthesis ---------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMdocumentation -------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMdocumentation% Start-check if style AMdocumentation is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMdocumentation.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%
\RequirePackage{longtable}%
\RequirePackage{fancyhdr}%
%
%
%=============
%--- Style ---
%=============
% Set default layout
\geometry{left=2.5cm,right=3cm,top=2.5cm,bottom=3cm,headsep=1cm,headheight=14pt}%
%
% Paragraph indentation
\AMlayoutSetIndent%
%
% Package titlesec: chapter and section style
\titlespacing*\chapter{0pt}{20pt plus 4pt minus 2pt}{10pt plus 2pt minus 2pt}%
\titlespacing*\section{0pt}{15pt plus 4pt minus 2pt}{7pt plus 2pt minus 2pt}%
\titlespacing*\subsection{0pt}{10pt plus 2pt minus 2pt}{2pt plus 2pt minus 2pt}%
\titlespacing*\subsubsection{0pt}{10pt plus 2pt minus 2pt}{2pt plus 2pt minus 2pt}%
\titleformat{\chapter}[display]{\color{TUMBlue}\sffamily\huge\bfseries}{\chaptertitlename\ \thechapter}{1em}{}%
\titleformat{\section}{\color{TUMBlue}\sffamily\Large\bfseries}{\thesection}{1em}{}%
\titleformat{\subsection}{\color{TUMBlue}\sffamily\large\bfseries}{\thesubsection}{1em}{}%
\titleformat{\subsubsection}{\color{TUMBlue}\sffamily\normalsize\bfseries}{\thesubsubsection}{1em}{}%
%
%% Package fancyhdr: header and footer style
\fancypagestyle{AMdocumentationp}{%
    \AtBeginDocument{%
	    \renewcommand{\sectionmark}[1]{\markright{\thesection\ \ ##1}}%
	    \renewcommand{\chaptermark}[1]{\markboth{\thechapter\ \ ##1}{}}%
    }%
    \fancyhf{}%
    \fancyhead[R]{\normalcolor\normalsize\bfseries\sffamily\thepage}%
    \fancyhead[L]{\normalcolor\normalsize\itshape\sffamily\nouppercase{\rightmark}}%
    \renewcommand{\headrulewidth}{0pt}%
    \renewcommand{\footrulewidth}{0pt}%
}%
%
% Set pagestyle for mainmatter
\pagestyle{AMdocumentationp}%
%
% First page of chapter empty instead of plain style
\patchcmd{\chapter}{plain}{empty}{}{}%
%
% Hyperref style
\hypersetup{linkcolor=TUMBlue,urlcolor=TUMBlue}%
%
% Caption style
\captionsetup{labelfont=bf,font={sf,small}}%
%
%
%================
%--- Commands ---
%================
\newcommand{\AMdocumentationModule}[1]{\texttt{\color{TUMGreen}#1}}%
\newcommand{\AMdocumentationStyle}[1]{\texttt{\color{TUMOrange}#1}}%
\newcommand{\AMdocumentationTemplate}[1]{\texttt{\color{TUMGray2}#1}}%
\newcommand{\AMdocumentationOption}[1]{\texttt{\color{TUMGray2}#1}}%
\newcommand{\AMdocumentationCode}[1]{\texttt{#1}}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMdocumentation.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if style AMdocumentation is enabled
%###################################################################################################
%--- END STYLE AMdocumentation ---------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMlecturescript -------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMlecturescript% Start-check if style AMlecturescript is enabled
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMlecturescript.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%      Setup of caption headers
\RequirePackage{fancyhdr}%     Setup of header and footer
\RequirePackage[nottoc]{tocbibind}%
\RequirePackage{multirow}%
\RequirePackage{booktabs}%
\RequirePackage{fancybox}%
\RequirePackage{listings}%     Typeset code
\RequirePackage{tabularx}%
\RequirePackage{empheq}%
\RequirePackage{tcolorbox}%    Colored and framed text boxes
\tcbuselibrary{skins}%         Load skin library
%
%
%=============
%--- Style ---
%=============
% Set default layout
\geometry{left=2.3cm,right=2.3cm,top=2cm,bottom=2.3cm,headheight=18pt}%
%
% Paragraph indentation
\AMlayoutSetIndent%
%
% Package titlesec: chapter and section style
\titlespacing*\chapter{0pt}{50pt plus 4pt minus 2pt}{40pt plus 2pt minus 2pt}%
\titlespacing*\section{0pt}{30pt plus 4pt minus 2pt}{16pt plus 2pt minus 2pt}%
\titlespacing*\subsection{0pt}{26pt plus 4pt minus 2pt}{14pt plus 2pt minus 2pt}%
\titlespacing*\subsubsection{0pt}{20pt plus 4pt minus 2pt}{6pt plus 2pt minus 2pt}%
\titleformat{\chapter}[display]{\color{TUMBlue}\sffamily\huge\bfseries}{\large\chaptertitlename\ \thechapter}{0.3em}{}%
\titleformat{\section}{\color{TUMBlue}\sffamily\Large\bfseries}{\thesection}{1em}{}%
\titleformat{\subsection}{\color{TUMBlue}\sffamily\large\bfseries}{\thesubsection}{1em}{}%
\titleformat{\subsubsection}{\color{TUMBlue}\sffamily\normalsize\bfseries}{\thesubsubsection}{1em}{}%
%
% Package fancyhdr: header and footer style
\fancypagestyle{AMlecturescriptp}{%
	\AtBeginDocument{%
		\renewcommand{\sectionmark}[1]{\markright{\thesection\ \ ##1}}%
		\renewcommand{\chaptermark}[1]{\markboth{\thechapter\ \ ##1}{}}%
	}%
	\fancyhf{}%
	\fancyhead[LE,RO]{\normalcolor\large\sffamily\thepage}%
	\fancyhead[LO]{\normalcolor\normalsize\sffamily\nouppercase{\rightmark}}%
	\fancyhead[RE]{\normalcolor\normalsize\sffamily\nouppercase{\leftmark}}%
	\renewcommand{\headrulewidth}{0pt}%
	\renewcommand{\footrulewidth}{0pt}%
}%
%
% Set numbering scheme
\numberwithin{equation}{section}%
%
% Set pagestyle for mainmatter
\pagestyle{AMlecturescriptp}%
%
% Change plain style
\fancypagestyle{plain}{%
	\fancyhf{}% clear all header and footer fields
	\fancyfoot[C]{\normalcolor\large\sffamily\thepage}%
	\renewcommand{\headrulewidth}{0pt}%
	\renewcommand{\footrulewidth}{0pt}}%
%
% Hyperref style (no color)
\hypersetup{linkcolor=black,urlcolor=black}%
%
% Caption style
\captionsetup{labelfont=bf,font={sf,small}}%
%
% Emphasized equations style
\newcommand{\empheqinnerbox}[1]{%
  \colorlet{currentcolor}{.}%
  {%
    \color{TUMBlue}%
    \fboxsep=1ex\fbox{\color{currentcolor}#1}%
  }%
}%
\empheqset{innerbox=\empheqinnerbox}%
%
%
%================
%--- Commands ---
%================
% Titlepage
% ---------
\newcommand{\AMlecturescriptTitlePage}[6]{{%
	\thispagestyle{empty}%
    \phantomsection%
    \addcontentsline{toc}{chapter}{\AMlangGerEng{Titelblatt}{Title Page}}%
	\AMlayoutHeaderCustomChair%
    \vspace*{1.5cm}%
    \begin{center}%
        {\sffamily\bfseries\Huge #1}\\[1cm]%
        {\sffamily\large\AMlangGerEng{Modul}{Module} #2}\\[1em]%
        \AMlangGerEng{Skript zur Vorlesung}{Lecture Script} - Version #3\\[0.3em]%
        #4%
    \end{center}%
    \vspace*{0.5cm}%
    \begin{center}%
        #6%
    \end{center}%
    \vfill%
    \begin{center}%
        {\large #5}\\[0.5em]%
        \AMlangTUM\\%
        \AMlangChair%
    \end{center}%
	\cleardoublepage%
}}%
%
% Contents/Figures/Tables
% -----------------------
\newcommand{\AMlecturescriptPrintTableOfContents}{%
	\AMlayoutNoIndent%
	\tableofcontents%
	\AMlayoutSetIndent%
	\cleardoublepage%
}%
%
\newcommand{\AMlecturescriptStartMiniToc}[1][2]{%
    \startcontents%   Starts counting sections...
    \vspace*{-1.5em}%
    \begingroup%
        \small%
        \noindent\textbf{\sffamily \AMlangGerEng{Inhalt}{Contents}:}%
        \printcontents{l}{1}[#1]{}%
    \endgroup%
    \vspace*{1.5em}%
	\noindent%
}%
%
\newcommand{\AMlecturescriptStopMiniToc}{%
    \stopcontents%   Stops counting sections...
}%
%
% Misc
% ----
\newtcolorbox{AMhighlight}[1][]{empty,fontupper=\itshape,coltext=TUMBlue}%
%
\newcounter{AMtheoremCounter}%
\counterwithin{AMtheoremCounter}{chapter}
\newenvironment{AMtheorem}[1][]{%
	\refstepcounter{AMtheoremCounter}%
	\par%
	\vspace{1em}%
	\leftskip=1em%
	\rightskip=\leftskip%
	\color{TUMBlue}%
	{\bfseries\AMlangGerEng{Theorem}{Theorem} \theAMtheoremCounter%
		\ifx\\#1\\\else%    %check, if argument is not empty
			\noindent\ -- #1%
		\fi%
		:}%
	\itshape%
	\quad%
}{%
	\par%
	\vspace{1em}%
}%
\crefname{AMtheoremCounter}{\AMlangGerEng{Theorem}{theorem}}{\AMlangGerEng{Theoreme}{theorems}}%
\Crefname{AMtheoremCounter}{\AMlangGerEng{Theorem}{Theorem}}{\AMlangGerEng{Theoreme}{Theorems}}%
%
\newenvironment{AMtheorem*}[1][]{%
	\par%
	\vspace{1em}%
	\leftskip=1em%
	\rightskip=\leftskip%
	\color{TUMBlue}%
	{\bfseries\AMlangGerEng{Theorem}{Theorem}%
		\ifx\\#1\\\else%    %check, if argument is not empty
		\noindent\ -- #1%
		\fi%
		:}%
	\itshape%
	\quad%
}{%
	\par%
	\vspace{1em}
}%
%
%
\newcounter{AMremarkCounter}%
\counterwithin{AMremarkCounter}{chapter}
\newenvironment{AMremark}[1][]{%
	\refstepcounter{AMremarkCounter}%
	\par%
	\vspace{1em}%
	\leftskip=1em%
	\rightskip=\leftskip%
	\color{TUMGray1}%
	\small%
	{\bfseries\AMlangGerEng{Bemerkung}{Remark} \theAMremarkCounter%
		\ifx\\#1\\\else%    %check, if argument is not empty
			\noindent\ -- #1%
		\fi%
		:}%
	\itshape%
	\quad%
}{%
	\par%
	\vspace{1em}%
}%
\crefname{AMremarkCounter}{\AMlangGerEng{Bemerkung}{remark}}{\AMlangGerEng{Bemerkungen}{remarks}}%
\Crefname{AMremarkCounter}{\AMlangGerEng{Bemerkung}{Remark}}{\AMlangGerEng{Bemerkungen}{Remarks}}%
%
\newenvironment{AMremark*}[1][]{%
	\par%
	\vspace{1em}
	\leftskip=1em%
	\rightskip=\leftskip%
	\color{TUMGray1}%
	\small%
	{\bfseries\AMlangGerEng{Bemerkung}{Remark}
		\ifx\\#1\\\else%    %check, if argument is not empty
		\noindent\ -- #1%
		\fi%
		:}%
	\itshape%
	\quad%
}{%
	\par%
	\vspace{1em}
}%
%
%
\newcounter{AMexampleCounter}%
\counterwithin{AMexampleCounter}{chapter}
\newenvironment{AMexample}[1][]{%
    \refstepcounter{AMexampleCounter}%
    \small%
    \paragraph*{%
      {\sffamily\color{TUMBlue}\AMlangGerEng{Beispiel}{Example} \theAMexampleCounter :}%
      \ifx\\#1\\\else%    %check, if argument is not empty
        \noindent\sffamily\ #1%
      \fi%
    }%
    \begin{quotation}%
}{%
    \end{quotation}%
}%
\crefname{AMexampleCounter}{\AMlangGerEng{Beispiel}{example}}{\AMlangGerEng{Beispiele}{examples}}%
\Crefname{AMexampleCounter}{\AMlangGerEng{Beispiel}{Example}}{\AMlangGerEng{Beispiele}{Examples}}%
%
\newenvironment{AMexample*}[1][]{%
	\small%
	\paragraph*{%
		{\sffamily\color{TUMBlue}\AMlangGerEng{Beispiel}{Example}:}%
		\ifx\\#1\\\else%    %check, if argument is not empty
		\noindent\sffamily\ #1%
		\fi%
	}%
	\begin{quotation}%
	}{%
	\end{quotation}%
}%
%
%
\newenvironment{AMexercise}[1][]{%
    \section*{\AMlangGerEng{Übungsaufgabe}{Exercise}}%
    \small%
    \begin{enumerate}%
}{%
    \end{enumerate}%
}%
%
%
%
\else\fi% End-check if style AMlecturescript is enabled
%###################################################################################################
%--- END STYLE AMlecturescript ---------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMphdexpose -----------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMphdexpose% Start-check if style AMphdexpose is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMphdexpose.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{caption}%
\RequirePackage{fancyhdr}%
%
%=============
%--- Style ---
%=============
% Set default layout
\geometry{left=3cm,right=3cm,top=2.5cm,bottom=3cm,headheight=14pt}%
%
% Paragraph indentation
\AMlayoutSetIndent%
%
% Package titlesec: section style
\titlespacing*\section{0pt}{30pt plus 4pt minus 2pt}{16pt plus 2pt minus 2pt}%
\titlespacing*\subsection{0pt}{20pt plus 4pt minus 2pt}{12pt plus 2pt minus 2pt}%
\titlespacing*\subsubsection{0pt}{10pt plus 4pt minus 2pt}{2pt plus 2pt minus 2pt}%
\titleformat{\section}{\color{TUMBlack}\sffamily\Large\bfseries}{\thesection}{1em}{}%
\titleformat{\subsection}{\color{TUMBlack}\sffamily\large\bfseries}{\thesubsection}{1em}{}%
\titleformat{\subsubsection}{\color{TUMBlack}\sffamily\normalsize\bfseries}{\thesubsubsection}{1em}{}%
%
% Package fancyhdr: header and footer style
\def\AMfooterAuthor{Author}%
\def\AMfooterDate{Date}%
\fancypagestyle{AMphdexposep}{%
	\AtBeginDocument{%
		\renewcommand{\sectionmark}[1]{\markright{\thesection\ \ ##1}}%
	}%
	\fancyhf{}%
	\fancyhead[L,R]{}%
	\fancyfoot[L]{\textit{\sffamily\footnotesize\AMfooterAuthor, \AMlangChair, \AMfooterDate}}%
	\fancyfoot[R]{\textit{\sffamily\thepage}}%
	\renewcommand{\headrulewidth}{0pt}%
	\renewcommand{\footrulewidth}{0pt}%
}%
%
% Set pagestyle
\pagestyle{AMphdexposep}%
%
% Change plain style
\fancypagestyle{plain}{%
	\fancyhf{} % clear all header and footer fields
	\fancyfoot[C]{\normalcolor\large\bfseries\sffamily\thepage}%
	\renewcommand{\headrulewidth}{0pt}%
	\renewcommand{\footrulewidth}{0pt}%
}%
%
% Caption style
\captionsetup{labelfont=bf,font={sf,small}}%
%
%
%=================
%--- Titlepage ---
%=================
\newcommand{\AMphdexposeTitlePage}[7]{%
	\def\AMfooterAuthor{#1}%
	\def\AMfooterDate{#7}%
    \AMrefSetPDFMetadata{Expos\'{e} \AMlangGerEng{für das Promotionsvorhaben}{for the Doctoral Project}}{}{#1}{}{}{}%
    \pagenumbering{roman}%
    \thispagestyle{empty}%
    \AMlayoutHeaderCDSchoolTUMLogoOnly%
    \addcontentsline{toc}{section}{\AMlangGerEng{Titelblatt}{Titlepage}}%
    \vspace*{3cm}%
    \begingroup%
        \sffamily%
        \begin{center}%
            \Large \textbf{Expos\'{e} \AMlangGerEng{für das Promotionsvorhaben}{for the Doctoral Project}}\\[1cm]%
            \large \textit{\AMlangGerEng{Vorläufiger Titel}{Provisional Subject}:}\\%
            \huge \textbf{\color{TUMBlue} #4}\\[0.5em]%
            \Large #1\\[0.5em]%
            \normalsize #7%
        \end{center}%
        \vfill%
		\noindent%
        \begin{tabular}{rl}%
            Name: & #1 \\%
            \AMlangGerEng{Anschrift}{Address}: & \AMlangTUM\\%
             & \AMlangSchool\\%
             & \AMlangChair\\%
             & \AMaddressStreetChair\\%
             & \AMaddressPostalCodeChair\ \AMaddressCityChair\\%
            E-Mail: & #2\\%
            \AMlangGerEng{Telefon}{Phone}: & #3\\[2em]%
            \AMlangSupervisor: & #5\\%
            Mentor: & #6%
        \end{tabular}%
    \endgroup%
    %
    \clearpage%
    \pagenumbering{arabic}%
}%
%
%==================
%--- Backmatter ---
%==================
% End document according to TUM Graduate School template
\AtEndDocument{%
    \clearpage%
    \thispagestyle{empty}%
    \section*{\AMlangGerEng{Überprüfung}{Review}}%
    \begingroup%
        \sffamily%
        \textit{\AMlangGerEng{Inhaltliche Ausführung erstellt von}{Content presentation created by}:}\\%
        \begin{tabular}{@{}p{0.5\textwidth}@{}p{0.5\textwidth}@{}}%
             & \\[3mm]%
            \rule{3cm}{0.3mm}\;, \;\rule{3cm}{0.3mm} & \rule{3cm}{0.3mm}\;, \;\rule{3cm}{0.3mm}\\%
            \textit{\AMlangGerEng{Ort, Datum}{Place, date}} & \textit{\AMlangGerEng{Ort, Datum}{Place, date}}\\[8mm]%
            \rule{6cm}{0.3mm} & \rule{6cm}{0.3mm}\\%
            \AMlangGerEng{Doktorand/in}{Doctoral candidate} & \AMlangGerEng{Erstprüfer/in}{First examiner}%
        \end{tabular}\par%
    \endgroup%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMphdexpose.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
%
\else\fi% End-check if style AMphdexpose is enabled
%###################################################################################################
%--- END STYLE AMphdexpose -------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMletter --------------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMletter% Start-check if style AMletter is enabled
%
% >>> CONTENTS OF FILE source/TUM/Styles/TUMletter.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%================
%--- Packages ---
%================
\RequirePackage{lastpage}%
%
%
%=============
%--- Style ---
%=============
% Document setup
\geometry{a4paper,left=2.5cm,right=2cm,top=3.5cm,bottom=3.5cm}%
\pagestyle{empty}%
\AMfontSetDocumentFontSizeXII%
%
% Paragraph indentation
\parindent0pt%
\parskip1.5em%
%
%
%================
%--- Commands ---
%================
\newcommand{\AMletterSetReceiver}[1]{%
    \def\AMletterReceiver{\begin{tabular}{@{}l}#1\end{tabular}}%
}%
\AMletterSetReceiver{Fa. Muster Mechanik\\Herrn Max Mustermann\\Musterstraße 12\\12345 Musterstadt}% Initialization
%
\newcommand{\AMletterSetPlace}[1]{%
    \def\AMletterPlace{#1}%
}%
\AMletterSetPlace{\AMaddressCityChair}% Initialization
%
\newcommand{\AMletterSetDate}[1]{%
    \def\AMletterDate{#1}%
}%
\AMletterSetDate{\AMutilsToday}% Initialization
%
\newcommand{\AMletterSetSubject}[1]{%
    \def\AMletterSubject{\textbf{#1}}%
}%
\AMletterSetSubject{\AMlangGerEng{Betreff}{Subject}}% Initialization
%
\newcommand{\AMletterSetTitle}[1]{%
    \def\AMletterTitle{#1}%
}%
\AMletterSetTitle{M.Sc.}% Initialization
%
\newcommand{\AMletterSetName}[1]{%
    \def\AMletterName{#1}%
}%
\AMletterSetName{Max Mustermann}% Initialization
%
\newcommand{\AMletterSetPhone}[1]{%
    \def\AMletterPhone{#1}%
}%
\AMletterSetPhone{+49 89 289 xxxxx}% Initialization
%
\newcommand{\AMletterSetFax}[1]{%
    \def\AMletterFax{#1}%
}%
\AMletterSetFax{+49 89 289 xxxxx}% Initialization
%
\newcommand{\AMletterSetEMail}[1]{%
    \def\AMletterEMail{#1}%
}%
\AMletterSetEMail{max.mustermann@tum.de}% Initialization
%
\newcommand{\AMletterSetHomepage}[1]{%
    \def\AMletterHomepage{#1}%
}%
\AMletterSetHomepage{\AMaddressWebsiteChair}% Initialization
%
\newcommand{\AMletterHeader}{%
    % Page background
    \AMlayoutPutAtNorthEast{2cm}{2cm}{\AMlogoTUM{1cm}{TUMBlue}}%
    \AddToHook{shipout/background}{\AMlayoutPutAtNorthWest{7mm}{87mm}{\tikz \draw[line width=0.1mm,TUMBlack] (0,0) -- (0.25,0);}}%
    \AddToHook{shipout/background}{\AMlayoutPutAtNorthWest{7mm}{148.5mm}{\tikz \draw[line width=0.1mm,TUMBlack] (0,0) -- (0.25,0);}}%
    \AddToHook{shipout/background}{\AMlayoutPutAtNorthWest{7mm}{192mm}{\tikz \draw[line width=0.1mm,TUMBlack] (0,0) -- (0.25,0);}}%
    \AddToHook{shipout/background}{\AMlayoutPutAtSouthWest{2.5cm}{1cm}{%
        \begingroup%
            \fontsize{9}{9}\selectfont%
            \noindent%
            \begin{tabular}{llll}%
                \textbf{\AMlangTUM} & \textbf{\AMletterTitle} & Tel. \AMletterPhone & \AMaddressCreditInstitutionChair\\%
                \AMlangSchool & \textbf{\AMletterName} & Fax \AMletterFax & IBAN-Nr.:\\%
                \AMlangChair & \AMaddressStreetChair & & \AMaddressIBANChair\\%
                 & \AMaddressPostalCodeChair\ \AMaddressCityChair & \AMletterEMail & BIC: \AMaddressBICChair\\%
                 & & \AMletterHomepage & Steuer-Nr.: \AMaddressTaxNumberChair\\
                 & & \AMaddressWebsiteTUM & USt-IdNr.: \AMaddressSalesTaxIDChair%
            \end{tabular}%
        \endgroup%
    }}%
    \AddToHook{shipout/background}{\AMlayoutPutAtSouthEast{0.8cm}{1cm}{{\fontsize{9}{9}\selectfont\thepage\ / \pageref*{LastPage}}}}%
    %
    % Header
    \begingroup%
        \fontsize{8}{11}\selectfont%
        \noindent%
        \begin{tabular}{@{}l}%
            \AMlangTUM\ | \AMaddressStreetChair\ | \AMaddressPostalCodeChair\ \AMaddressCityChair\\%
            \AMlangChair\ | \AMlangSchool%
        \end{tabular}%
    \endgroup%
    \\[0.2em]%
    \noindent\AMletterReceiver\par%
    \vspace*{3em}%
    \noindent\AMletterPlace, \AMletterDate\par%
    \noindent\AMletterSubject\par%
    \vspace*{2em}%
}%
%
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMletter.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%
%
\newcommand{\AMletterCustomHeader}{%
	% Page background
	\AMlayoutPutAtNorthEast{2cm}{2cm}{\AMlogoTUM{1cm}{TUMBlue}}%
	\AddToHook{shipout/background}{\AMlayoutPutAtNorthWest{7mm}{87mm}{\tikz \draw[line width=0.1mm,TUMBlack] (0,0) -- (0.25,0);}}%
	\AddToHook{shipout/background}{\AMlayoutPutAtNorthWest{7mm}{148.5mm}{\tikz \draw[line width=0.1mm,TUMBlack] (0,0) -- (0.25,0);}}%
	\AddToHook{shipout/background}{\AMlayoutPutAtNorthWest{7mm}{192mm}{\tikz \draw[line width=0.1mm,TUMBlack] (0,0) -- (0.25,0);}}%
	\AddToHook{shipout/background}{\AMlayoutPutAtSouthWest{2.5cm}{1cm}{%
			\begingroup%
			\fontsize{9}{9}\selectfont%
			\noindent%
			\begin{tabular}{llll}%
				\textbf{\AMlangTUM} & \textbf{\AMletterTitle} & Tel. \AMletterPhone & \AMaddressCreditInstitutionChair\\%
				\AMlangSchool & \textbf{\AMletterName} & & IBAN-Nr.:\\%
				\AMlangChair & \AMaddressStreetChair & \AMletterEMail & \AMaddressIBANChair\\%
				& \AMaddressPostalCodeChair\ \AMaddressCityChair & \AMletterHomepage & BIC: \AMaddressBICChair\\%
				& & \AMaddressWebsiteTUM & Steuer-Nr.: \AMaddressTaxNumberChair\\
				& &  & USt-IdNr.: \AMaddressSalesTaxIDChair%
			\end{tabular}%
			\endgroup%
	}}%
	\AddToHook{shipout/background}{\AMlayoutPutAtSouthEast{0.8cm}{1cm}{{\fontsize{9}{9}\selectfont\thepage\ / \pageref*{LastPage}}}}%
	%
	% Header
	\begingroup%
	\fontsize{8}{11}\selectfont%
	\noindent%
	\begin{tabular}{@{}l}%
		\AMlangTUM\ | \AMaddressStreetChair\ | \AMaddressPostalCodeChair\ \AMaddressCityChair\\%
		\AMlangChair\ | \AMlangSchool%
	\end{tabular}%
	\endgroup%
	\\[0.2em]%
	\noindent\AMletterReceiver\par%
	\vspace*{3em}%
	\noindent\AMletterPlace, \AMletterDate\par%
	\noindent\AMletterSubject\par%
	\vspace*{2em}%
}%
%
\else\fi% End-check if style AMletter is enabled
%###################################################################################################
%--- END STYLE AMletter ----------------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- BEGIN STYLE AMdisplayitem ---------------------------------------------------------------------
%###################################################################################################
\ifdefined\optAMdisplayitem% Start-check if style AMdisplayitem is enabled
%
% >>> CONTENTS OF FILE source/ED/AM/Styles/AMdisplayitem.sty: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
%=============
%--- Style ---
%=============
% PDF metadata
\hypersetup{%
    pdftoolbar = false,%
    pdfmenubar = false,%
    pdfsubject = {\AMlangChair, \AMlangTUM}%
}%
%
%
%==============
%--- Layout ---
%==============
\newcommand{\AMdisplayitemHeader}[6]{%
    % PDF metadata
    \hypersetup{%
        pdftitle = {#1},%
        pdfauthor = {#4}%
    }%
    %
    % Item-specific header
    \twocolumn[{%
        \begin{tikzpicture}[remember picture,overlay]%
            \node [xshift=6cm,yshift=-6cm] at (current page.north west){\includegraphics[width=0.7\columnwidth]{#6}};%
        \end{tikzpicture}%
        \begin{flushright}%
            \vspace{1em}%
            {\large\sffamily Lehrmodellbeschreibung}\\%
            \vspace{1em}%
            {\LARGE\bfseries\color{TUMBlue}\sffamily#1}\\%
            \vspace{1em}%
            \textbf{\large\sffamily #2}\\%
            \vspace{1em}%
            \sffamily Hersteller: #3\\%
            \vspace{.5em}%
            \sffamily Betreuer: #4\\%
            \vspace{.5em}%
            \large\sffamily #5%
        \end{flushright}%
        \vspace{2cm}%
    }]%
    %
    % Default header
    \AMlayoutHeaderCustomChair%
}%
%
%
%
\else\fi% End-check if style AMdisplayitem is enabled
%###################################################################################################
%--- END STYLE AMdisplayitem -----------------------------------------------------------------------
%###################################################################################################
%
%
%###################################################################################################
%--- UPDATING --------------------------------------------------------------------------------------
%###################################################################################################
\ifnum\year>2025%
    \def\AMlatexUpdateRecommended{x}%
\else%
    \ifnum\year=2025%
        \ifnum\month>3%
            \def\AMlatexUpdateRecommended{x}%
        \else%
             \ifnum\month=3%
                 \ifnum\day>21%
                     \def\AMlatexUpdateRecommended{x}%
                 \else%
                     \ifnum\day=21%
                         \def\AMlatexUpdateRecommended{x}%%
                     \fi%
                 \fi%
             \fi%
        \fi%
    \fi%
\fi%
\ifdefined\AMlatexUpdateRecommended%
    \AMcoreLogWarning{The AMlatex package you are using is more than one year old (build on 2024-3-21). Please consider an update!}%
\fi%
%
%
%###################################################################################################
%--- DEBUGGING -------------------------------------------------------------------------------------
%###################################################################################################
\newcommand{\AMdebug}{%
\newpage%
\noindent\textbf{Loaded modules:}\\%
AMcore: \ifdefined\optAMcore yes\else no\fi\\%
AMfont: \ifdefined\optAMfont yes\else no\fi\\%
AMlang: \ifdefined\optAMlang yes\else no\fi\\%
AMaddress: \ifdefined\optAMaddress yes\else no\fi\\%
AMcolor: \ifdefined\optAMcolor yes\else no\fi\\%
AMtikz: \ifdefined\optAMtikz yes\else no\fi\\%
AMlogo: \ifdefined\optAMlogo yes\else no\fi\\%
AMmath: \ifdefined\optAMmath yes\else no\fi\\%
AMlayout: \ifdefined\optAMlayout yes\else no\fi\\%
AMbiblio: \ifdefined\optAMbiblio yes\else no\fi\\%
AMnames: \ifdefined\optAMnames yes\else no\fi\\%
AMutils: \ifdefined\optAMutils yes\else no\fi\\%
AMref: \ifdefined\optAMref yes\else no\fi\\%
\noindent\textbf{Loaded styles:}\\%
AMstudentexpose: \ifdefined\optAMstudentexpose yes\else no\fi\\%
AMstudentthesiscover: \ifdefined\optAMstudentthesiscover yes\else no\fi\\%
AMbulletin: \ifdefined\optAMbulletin yes\else no\fi\\%
AMexercise: \ifdefined\optAMexercise yes\else no\fi\\%
AMdocument: \ifdefined\optAMdocument yes\else no\fi\\%
AMbeamer: \ifdefined\optAMbeamer yes\else no\fi\\%
AMposter: \ifdefined\optAMposter yes\else no\fi\\%
AMphdprepublication: \ifdefined\optAMphdprepublication yes\else no\fi\\%
AMphdthesis: \ifdefined\optAMphdthesis yes\else no\fi\\%
AMempty: \ifdefined\optAMempty yes\else no\fi\\%
AMstudentthesis: \ifdefined\optAMstudentthesis yes\else no\fi\\%
AMdocumentation: \ifdefined\optAMdocumentation yes\else no\fi\\%
AMlecturescript: \ifdefined\optAMlecturescript yes\else no\fi\\%
AMphdexpose: \ifdefined\optAMphdexpose yes\else no\fi\\%
AMletter: \ifdefined\optAMletter yes\else no\fi\\%
AMdisplayitem: \ifdefined\optAMdisplayitem yes\else no\fi\\%
\noindent\textbf{Selected options:}\\%
optHelvetica: \ifdefined\optHelvetica yes\else no\fi\\%
optArial: \ifdefined\optArial yes\else no\fi\\%
optCharter: \ifdefined\optCharter yes\else no\fi\\%
optComputerModern: \ifdefined\optComputerModern yes\else no\fi\\%
optCharterMath: \ifdefined\optCharterMath yes\else no\fi\\%
optComputerModernMath: \ifdefined\optComputerModernMath yes\else no\fi\\%
optGerman: \ifdefined\optGerman yes\else no\fi\\%
optEnglish: \ifdefined\optEnglish yes\else no\fi\\%
optRGB: \ifdefined\optRGB yes\else no\fi\\%
optCMYK: \ifdefined\optCMYK yes\else no\fi\\%
optGray: \ifdefined\optGray yes\else no\fi\\%
optMonochrome: \ifdefined\optMonochrome yes\else no\fi\\%
optTikzExternalize: \ifdefined\optTikzExternalize yes\else no\fi\\%
optLeftEquations: \ifdefined\optLeftEquations yes\else no\fi\\%
optCenterEquations: \ifdefined\optCenterEquations yes\else no\fi\\%
optLimitMathFonts: \ifdefined\optLimitMathFonts yes\else no\fi\\%
optAfive: \ifdefined\optAfive yes\else no\fi\\%
optAfour: \ifdefined\optAfour yes\else no\fi\\%
optAthree: \ifdefined\optAthree yes\else no\fi\\%
optAtwo: \ifdefined\optAtwo yes\else no\fi\\%
optAone: \ifdefined\optAone yes\else no\fi\\%
optAzero: \ifdefined\optAzero yes\else no\fi\\%
optBeamerClassicFormat: \ifdefined\optBeamerClassicFormat yes\else no\fi\\%
optBeamerWideFormat: \ifdefined\optBeamerWideFormat yes\else no\fi\\%
optExzellenz: \ifdefined\optExzellenz yes\else no\fi\\%
optBiber: \ifdefined\optBiber yes\else no\fi\\%
optBibtex: \ifdefined\optBibtex yes\else no\fi\\%
optBibstyleNumeric: \ifdefined\optBibstyleNumeric yes\else no\fi\\%
optBibstyleAlphabetic: \ifdefined\optBibstyleAlphabetic yes\else no\fi\\%
optBibstyleAuthorYear: \ifdefined\optBibstyleAuthorYear yes\else no\fi\\%
optHideTodos: \ifdefined\optHideTodos yes\else no\fi\\%
optHideAnnotations: \ifdefined\optHideAnnotations yes\else no\fi\\%
optBlackRefs: \ifdefined\optBlackRefs yes\else no\fi\\%
optCoverWithMarks: \ifdefined\optCoverWithMarks yes\else no\fi\\%
optCoverWithBorders: \ifdefined\optCoverWithBorders yes\else no\fi\\%
optStudentBulletin: \ifdefined\optStudentBulletin yes\else no\fi\\%
optResearchBulletin: \ifdefined\optResearchBulletin yes\else no\fi\\%
optTeachingBulletin: \ifdefined\optTeachingBulletin yes\else no\fi\\%
optBulletinCropPage: \ifdefined\optBulletinCropPage yes\else no\fi\\%
optLectureSlides: \ifdefined\optLectureSlides yes\else no\fi\\%
optExerciseSlides: \ifdefined\optExerciseSlides yes\else no\fi\\%
}%
%
%
\endinput%