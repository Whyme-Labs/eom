# JSON

JSON (JavaScript Object Notation) is an open standard file format and data interchange format that uses human-readable text to store and transmit data objects consisting of name-value pairs and arrays. It serves as a commonly used data format with diverse applications in electronic data interchange, particularly for web applications communicating with servers.

The format is programming language-independent, though it was derived from JavaScript. Most modern programming languages include built-in functionality to generate and parse JSON-format data. JSON filenames use the .json extension.

Douglas Crockford originally specified JSON in the early 2000s, with the first JSON message sent in April 2001. The format grew out of a practical need for real-time server-to-browser communication without relying on browser plugins like Flash or Java applets, which dominated the early 2000s.

## History

JSON emerged from work at State Software, a company cofounded by Crockford and others in March 2001. The founders designed a system that would use standard browser capabilities and provide an abstraction layer for developers to create stateful web applications with persistent duplex connections to web servers. They held a round-table discussion and voted on whether to call the format JSML (JavaScript Markup Language) or JSON, ultimately choosing the latter.

The JSON.org website launched in 2001. By December 2005, Yahoo! began offering some of its web services in JSON format, marking significant early adoption. The format was based on a subset of the JavaScript scripting language, specifically Standard ECMA-262 3rd Edition from December 1999.

Formal standardization came later. After RFC 4627 served as an informational specification since 2006, JSON was first formally standardized in 2013 as ECMA-404. RFC 8259, published in 2017, became the current version of Internet Standard STD 90. That same year, JSON was also standardized as ISO/IEC 21778:2017. The ECMA and ISO/IEC standards describe only the allowed syntax, while the RFC covers security and interoperability considerations.

## Syntax

The fundamental structure of JSON relies on simple, human-readable syntax. The format represents data using name-value pairs and ordered lists of values. JSON's basic data types include numbers (signed decimal numbers that may contain fractional parts and exponential notation), strings (sequences of Unicode characters delimited by double quotation marks), booleans (true or false values), arrays (ordered lists of zero or more elements), objects (collections of name-value pairs), and null values.

Objects are delimited with curly brackets and use colons to separate keys from values, with commas separating pairs. Arrays use square bracket notation with comma-separated elements. Whitespace is allowed and ignored around syntactic elements, though four specific characters qualify as whitespace: space, horizontal tab, line feed, and carriage return.

JSON intentionally excludes comment syntax. The creator explained this design decision by noting concerns that people would use comments to hold parsing directives, which would destroy interoperability. Additionally, JSON disallows trailing commas after the last value in a data structure, though this feature appears in JSON derivatives to improve usability.

Character encoding follows strict standards. JSON exchange in open ecosystems must be encoded in UTF-8, supporting the full Unicode character set including characters beyond the Basic Multilingual Plane. When escaped, characters outside this range must be written using UTF-16 surrogate pairs.

## Data Types and Values

JSON provides a minimal but sufficient set of data types. Numbers make no distinction between integer and floating-point values and may use exponential notation. However, JSON cannot represent non-numbers like NaN. The format is agnostic regarding numeric representation within programming languages, which can lead to portability issues when implementations treat equivalent numbers differently.

Strings support a backslash escaping syntax and can contain any Unicode character when unescaped or use UTF-16 surrogate pair notation when escaped. Boolean values consist simply of true or false. Arrays provide ordered collections of any type, while objects represent unordered collections of name-value pairs where names are always strings.

The specification places no restrictions on strings used as object names, does not require name uniqueness, and assigns no significance to member ordering. This flexibility can sometimes create interoperability challenges when different implementations handle duplicate names unpredictably or when applications incorrectly assign meaning to member order.

## Security Considerations

A common misconception holds that JSON is safe to pass directly to the JavaScript eval() function since JSON is often described as a subset of JavaScript. This assumption proved dangerous because certain valid JSON texts, specifically those containing Unicode line separators or paragraph separators, were not valid in older JavaScript implementations. The JSON.parse() function, added to the fifth edition of ECMAScript, provides the proper secure method for parsing JSON data.

Various JSON parser implementations have historically suffered from denial-of-service attacks and mass assignment vulnerabilities. Developers must use dedicated parsing functions rather than eval() to safely process JSON from untrusted sources.

## Limitations and Design Trade-offs

JSON's simplicity brings both strengths and constraints. The format deliberately lacks a comment syntax, forcing developers using JSON for configuration files to employ preprocessing tools if annotation is desired. The specification does not constrain the magnitude or precision of number literals, though widely-used JavaScript implementations store numbers as IEEE754 binary64 quantities, creating potential precision issues with very large or precise decimal numbers.

The specification allows objects with duplicate member names, though such data is problematic for interoperability. Similarly, while specifications place no limits on character encoding, nearly all implementations assume UTF-8, making this effectively required for reliable interchange.

Early JSON definitions required that valid text consist only of objects or arrays at the root level, but this restriction was dropped in later specifications, allowing any serialized value as a top-level JSON document.
