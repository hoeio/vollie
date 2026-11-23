// memo-12 rounds 14 and 15 (Ebony): llamadart's own chat-template renderer on each served file's template.
// *host*, no model. Input: a JSON {"models": [...], "<model>": {GGUF key: string value, ...}, "prompt": "..."}.
// Prints the rendered prompt's head and tail and the stop strings; writes rendered-<model>.txt and, with a short
// assistant turn in the history, rendered2-<model>.txt (to see what eos_token becomes in a second turn).
// Run from lab/app-c3:  dart run --packages=.dart_tool/package_config.json <this file> <meta.json> <out dir>
import 'dart:convert';
import 'dart:io';
import 'package:llamadart/src/core/template/chat_template_engine.dart';
import 'package:llamadart/llamadart.dart' show LlamaChatMessage, LlamaChatRole;

void main(List<String> a) {
  final m = jsonDecode(File(a[0]).readAsStringSync()) as Map<String, dynamic>;
  final prompt = m['prompt'] as String;
  for (final g in (m['models'] as List).cast<String>()) {
    final md = (m[g] as Map).map((k, v) => MapEntry(k as String, v as String));
    final r = ChatTemplateEngine.render(
      templateSource: md['tokenizer.chat_template'],
      messages: [LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt)],
      metadata: md, addAssistant: true, enableThinking: false);
    final p = r.prompt;
    print('$g | format ${r.format} | chars ${p.length} | head ${jsonEncode(p.substring(0, p.length < 70 ? p.length : 70))} | tail ${jsonEncode(p.substring(p.length - 48))} | stops ${jsonEncode(r.additionalStops)}');
    File('${a[1]}/rendered-$g.txt').writeAsStringSync(p);
    final r2 = ChatTemplateEngine.render(
      templateSource: md['tokenizer.chat_template'],
      messages: [
        LlamaChatMessage.fromText(role: LlamaChatRole.user, text: 'Hi'),
        LlamaChatMessage.fromText(role: LlamaChatRole.assistant, text: 'Hello.'),
        LlamaChatMessage.fromText(role: LlamaChatRole.user, text: 'Bye'),
      ],
      metadata: md, addAssistant: true, enableThinking: false);
    print('   two turns: ${jsonEncode(r2.prompt)}');
    File('${a[1]}/rendered2-$g.txt').writeAsStringSync(r2.prompt);
  }
}
